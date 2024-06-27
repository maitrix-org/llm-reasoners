from .constants import *
from tarski import fstrips as fs
from tarski import model
from tarski.fstrips.problem import create_fstrips_problem
from tarski.io.fstrips import print_init, print_goal, print_formula, print_atom
from tarski.fstrips import language
from tarski.syntax import land, top, VariableBinding, Interval
from tarski.syntax import sorts
from tarski.io.fstrips import FstripsWriter
from tarski.errors import UndefinedSort


# TODO: Add Conditional Effects
# TODO: Add increase effects

class ModelWriter(object):
    def __init__(self, model, domain_name="test_domain", problem_name="instance-1"):
        self.model_dict = model
        self.predicate_map = {}
        self.functions = {}
        self.variable_map = {}
        self.fstrips_problem = create_fstrips_problem(language(), problem_name, domain_name)
        sorts.attach_arithmetic_sorts(self.fstrips_problem.language)
        #        self.fstrips_problem.metric_ = ("minimize","(total-cost)")
        self.populate_fstrips_problem()

    def populate_fstrips_problem(self):
        self.fstrips_problem.plan_metric = self.model_dict[METRIC]
        self.create_hierarchy()
        self.create_predicates()
        self.add_constants()
        self.create_functions()
        self.write_init()
        self.write_goal()
        self.write_actions()

    def create_hierarchy(self):
        # print(self.fstrips_problem.language._sorts)
        # print(self.fstrips_problem.language.ancestor_sorts)
        imm_parents = self.model_dict[HIERARCHY][IMM_PARENT]
        for obj in imm_parents:
            try:
                sort = self.fstrips_problem.language.get_sort(obj[0])
            except UndefinedSort:
                if obj[0]=='number': #Make number also a builtin sort
                    parent = self.fstrips_problem.language.get_sort(obj[1])
                    new_sort = Interval(obj[0], self.fstrips_problem.language, parent.encode, parent.lower_bound, parent.upper_bound,builtin=True)
                    self.fstrips_problem.language.attach_sort(new_sort,parent)
                    continue
                elif obj[2] == 1:
                    parent = self.fstrips_problem.language.get_sort(obj[1])
                    # print(parent)
                    self.fstrips_problem.language.interval(obj[0], parent, parent.lower_bound, parent.upper_bound)
                    continue
                self.fstrips_problem.language.sort(obj[0], obj[1])
        # print(self.fstrips_problem.language._sorts)

    def create_predicates(self):
        predicates = self.model_dict[PREDICATES]
        for predicate in predicates:
            sorts = []
            for s in predicate[1]:
                try:
                    sort = self.fstrips_problem.language.get_sort(s)
                except UndefinedSort:
                    sort = self.fstrips_problem.language.sort(s)
                sorts.append(sort)
            pred_obj = self.fstrips_problem.language.predicate(predicate[0], *sorts)
            self.predicate_map[predicate[0]] = pred_obj

    def add_constants(self):
        constants = self.model_dict[CONSTANTS]
        for constant in constants:
            try:
                sort = self.fstrips_problem.language.get_sort(constant[1])
            except UndefinedSort:
                sort = self.fstrips_problem.language.sort(constant[1])

            self.fstrips_problem.language.constant(constant[0], sort)

    def create_functions(self):
        functions = self.model_dict[FUNCTIONS]
        for function in functions:
            sorts = []
            for s in function[1]:
                try:
                    sort = self.fstrips_problem.language.get_sort(s)
                except UndefinedSort:
                    sort = self.fstrips_problem.language.sort(s)
                sorts.append(sort)
            func_obj = self.fstrips_problem.language.function(function[0], *sorts)
            self.functions[function[0]] = func_obj

    def write_init(self):
        functions = self.model_dict[INSTANCE][INIT][FUNCTIONS]
        predicates = self.model_dict[INSTANCE][INIT][PREDICATES]

        for function in functions:
            # print(function, self.functions[function[0]], type(self.functions[function[0]].__call__()))
            self.fstrips_problem.init.set(self.functions[function[0]].__call__(), function[1][0], *[function[1][0]])
        for predicate in predicates:
            self.fstrips_problem.init.add(self.predicate_map[predicate[0]], *predicate[1])

    def get_goals(self, fluent_list):
        temp_model = model.create(self.fstrips_problem.language)
        if len(fluent_list) == 0:
            return top
        elif len(fluent_list) <= 1:
            temp_model.add(self.predicate_map[fluent_list[0][0]], *fluent_list[0][1])
            return land(*temp_model.as_atoms())

        else:
            try:
                for subgoal in fluent_list:
                    temp_model.add(self.predicate_map[subgoal[0]], *subgoal[1])
                return land(*temp_model.as_atoms(), flat=True)
            except AssertionError as exc:
                raise Exception("Message:", exc, " Original fluent set", fluent_list)

    def write_goal(self):
        goal = self.model_dict[INSTANCE][GOAL]
        self.fstrips_problem.goal = self.get_goals(goal)

    # ACTIONS
    def get_conjunctions(self, act, fluent_list, flag):
        if len(fluent_list) == 0:
            if flag == POS_PREC:
                return top
            else:
                return []
        elif len(fluent_list) <= 1:
            fluent = fluent_list[0]
            variables = fluent[1]
            var = [self.variable_map[act][variable.replace('?', '')] for variable in variables]
            if flag == POS_PREC:
                return self.predicate_map[fluent[0]](*var)
            elif flag == ADDS:
                return [fs.AddEffect(self.predicate_map[fluent[0]](*var))]
            elif flag == DELS:
                return [fs.DelEffect(self.predicate_map[fluent[0]](*var))]
        else:
            and_fluent_list = []
            if flag == POS_PREC:
                for fluent in fluent_list:
                    variables = fluent[1]
                    var = [self.variable_map[act][variable.replace('?', '')] for variable in variables]
                    and_fluent_list.append(self.predicate_map[fluent[0]](*var))
                return land(*and_fluent_list, flat=True)
            elif flag == ADDS:
                for fluent in fluent_list:
                    variables = fluent[1]
                    var = [self.variable_map[act][variable.replace('?', '')] for variable in variables]
                    and_fluent_list.append(fs.AddEffect(self.predicate_map[fluent[0]](*var)))
                return and_fluent_list
            elif flag == DELS:
                for fluent in fluent_list:
                    variables = fluent[1]
                    var = [self.variable_map[act][variable.replace('?', '')] for variable in variables]
                    and_fluent_list.append(fs.DelEffect(self.predicate_map[fluent[0]](*var)))
                return and_fluent_list

    def write_actions(self):
        for act in self.model_dict[DOMAIN]:
            cost = self.model_dict[DOMAIN][act][COST]
            self.variable_map[act] = {}
            if PARARMETERS in self.model_dict[DOMAIN][act]:
                pars = []
                for p, s in self.model_dict[DOMAIN][act][PARARMETERS]:
                    try:
                        sort = self.fstrips_problem.language.get_sort(s)
                    except UndefinedSort:
                        sort = self.fstrips_problem.language.sort(s)
                    new_var = self.fstrips_problem.language.variable(p, sort)
                    
                    # ADDING VARIABLE TO VARIABLE MAP
                    if new_var.symbol in self.variable_map[act].keys():
                        pars.append(new_var)
                    else:
                        self.variable_map[act][new_var.symbol] = new_var
                        pars.append(new_var)
                precond = self.get_conjunctions(act, self.model_dict[DOMAIN][act][POS_PREC], POS_PREC)
                add_effects = self.get_conjunctions(act, self.model_dict[DOMAIN][act].get(ADDS, set()), ADDS)
                delete_effects = self.get_conjunctions(act, self.model_dict[DOMAIN][act].get(DELS, set()), DELS)
            else:
                pars = []
            self.fstrips_problem.action(act, pars, precond, add_effects + delete_effects, cost)

    def write_files(self, domain_file, problem_file):
        curr_writer = FstripsWriter(self.fstrips_problem)
        curr_writer.write(domain_file, problem_file)
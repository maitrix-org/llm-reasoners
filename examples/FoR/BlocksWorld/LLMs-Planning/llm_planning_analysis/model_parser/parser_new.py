import sys
import tarski
import tarski.io
from tarski.io.fstrips import print_init, print_goal, print_formula, print_atom
from tarski.syntax import CompoundFormula, formulas, Tautology, Atom
from tarski.syntax.terms import CompoundTerm, Constant
from tarski.syntax.sorts import Interval
from tarski.fstrips import AddEffect, DelEffect
from tarski.fstrips.fstrips import FunctionalEffect, IncreaseEffect
from .constants import *

#TODO: Negative Preconditions!!
#TODO: Conditional Effects testing.



def parse_model(domain_file, problem_file):
    reader = tarski.io.FstripsReader()
    reader.read_problem(domain_file,problem_file)
    model_dict = store_model(reader)
    return model_dict

def store_model(reader):
    model_dict = {}
    model_dict[METRIC] = reader.problem.plan_metric
    model_dict[PREDICATES] = store_predicates(reader)
    model_dict[FUNCTIONS] = store_functions(reader)
    model_dict[INSTANCE] = {}
    model_dict[INSTANCE][INIT] = {}
    model_dict[INSTANCE][INIT][FUNCTIONS], model_dict[INSTANCE][INIT][PREDICATES] = store_init(reader)
    model_dict[INSTANCE][GOAL] = store_goal(reader)
    model_dict[DOMAIN] = store_actions(reader)
    model_dict[HIERARCHY] = {}
    model_dict[HIERARCHY][ANCESTORS], model_dict[HIERARCHY][IMM_PARENT] = store_hierarchy(reader)
    model_dict[CONSTANTS] = store_constants(reader)
    return model_dict

def store_predicates(reader):
    predicates = list(reader.problem.language.predicates)
    predicates_list = []
    for preds in predicates:
        if str(preds.symbol) in ['=','!=','<','<=','>','>=']:
            continue
        predicates_list.append([preds.symbol,[sorts.name for sorts in preds.sort]])
    return predicates_list
def store_constants(reader):
    constants = reader.problem.language.constants()
    constant_list = []
    for constant in constants:
        constant_list.append([constant.symbol,constant.sort.name])
    return constant_list
def store_functions(reader):
    functions = list(reader.problem.language.functions)
    functions_list = []
    for funcs in functions:
        if str(funcs.symbol) in ['ite','@','+','-','*','/','**','%','sqrt','number']:
            continue
        functions_list.append([funcs.symbol,[sorts.name for sorts in funcs.sort]])
    return functions_list
def store_init(reader):
    inits = reader.problem.init.as_atoms()
    init_dict = {}
    init_dict[FUNCTIONS] = []
    init_dict[PREDICATES] = []
    for i in range(len(inits)):
        if not isinstance(inits[i],Atom):
            init_dict[FUNCTIONS].append([inits[i][0].symbol.symbol,[inits[i][1].symbol]])
        else:
            if len(inits[i].subterms) == 0:
                init_dict[PREDICATES].append([inits[i].symbol.symbol, []])
            else:
                init_dict[PREDICATES].append([inits[i].symbol.symbol, [subt.symbol for subt in inits[i].subterms]])

    return init_dict[FUNCTIONS], init_dict[PREDICATES]

def store_goal(reader):
    goal = reader.problem.goal
    goals = []
    if isinstance(goal,Tautology):
        goals.append([goal.symbol.symbol,[]])
    elif isinstance(goal,Atom):
        goals.append([goal.symbol.symbol,[subt.symbol for subt in goal.subterms]])
    else:
        for subformula in goal.subformulas:
            goals.append([subformula.symbol.symbol, [i.symbol for i in subformula.subterms]])
    return goals
def store_actions(reader):
    action_model = {}

    for act in reader.problem.actions.values():
        action_model[act.name] = {}
        # Add parameter list
        action_model[act.name][PARARMETERS] = [(p.symbol.replace('?',''), p.sort.name) for p in act.parameters]
        if isinstance(act.precondition, CompoundFormula):
            action_model[act.name][POS_PREC] = [[subformula.symbol.symbol,[i.symbol for i in subformula.subterms]] for subformula in act.precondition.subformulas]
        elif isinstance(act.precondition, formulas.Atom):
            action_model[act.name][POS_PREC] = [[act.precondition.symbol.symbol, [i.symbol for i in act.precondition.subterms]]]
        else:
            action_model[act.name][POS_PREC] = []
        action_model[act.name][ADDS] = []
        action_model[act.name][DELS] = []
        action_model[act.name][FUNCTIONAL] = []
        action_model[act.name][COND_ADDS] = []
        action_model[act.name][COND_DELS] = []
        action_model[act.name][COST] = act.cost
        for curr_effs in act.effects:
            if type(curr_effs) != list:
                curr_effs = [curr_effs]
            for eff in curr_effs:
                #Check if not tautology
                if not isinstance(eff.condition, Tautology):
                    # Conditional effects should be of the form [[condition,eff]]
                    curr_condition = []
                    if isinstance(eff.condition, CompoundFormula):
                        curr_condition.append([[subformula.symbol.symbol, [i.symbol for i in subformula.subterms]] for subformula in eff.condition.subformulas])
                    elif isinstance(eff.condition, Atom):
                        curr_condition.append([[eff.condition.symbol.symbol, [i.symbol for i in eff.condition.subterms]]])
                    if isinstance(eff, AddEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][COND_ADDS].append([curr_condition,[eff.atom.symbol.symbol, []]])
                        else:
                            action_model[act.name][COND_ADDS].append([curr_condition,[eff.atom.symbol.symbol, [subt.symbol for subt in eff.atom.subterms]]])
                    elif isinstance(eff, DelEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][COND_DELS].append([curr_condition,[eff.atom.symbol.symbol, []]])
                        else:
                            action_model[act.name][COND_DELS].append([curr_condition,[eff.atom.symbol.symbol, [subt.symbol for subt in eff.atom.subterms]]])
                    elif isinstance(eff, FunctionalEffect):
                        if "+" in str(eff.condition.symbol):
                            if(type(eff.rhs) is CompoundTerm):
                                action_model[act.name][FUNCTIONAL].append([[eff.lhs.symbol.symbol,eff.lhs.sort.name],[eff.rhs.symbol.symbol,eff.rhs.sort.name]])
                            elif(type(eff.rhs) is Constant):
                                action_model[act.name][FUNCTIONAL].append([[eff.lhs.symbol.symbol, eff.lhs.sort.name],[eff.rhs.symbol, eff.rhs.sort.name]])

                else:
                    if isinstance(eff, AddEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][ADDS].append([eff.atom.symbol.symbol, []])
                        else:
                            action_model[act.name][ADDS].append([eff.atom.symbol.symbol, [subt.symbol for subt in eff.atom.subterms]])
                    if isinstance(eff, DelEffect):
                        if len(eff.atom.subterms) == 0:
                            action_model[act.name][DELS].append([eff.atom.symbol.symbol, []])
                        else:
                            action_model[act.name][DELS].append([eff.atom.symbol.symbol, [subt.symbol for subt in eff.atom.subterms]])


    return action_model
def store_hierarchy(reader):
    ancestors = reader.problem.language.ancestor_sorts
    ancestor_list = []
    for key,value in ancestors.items():
        if len(value)==0:
            ancestor_list.append([key.name,[],int(type(key)==Interval)])
            ancestor_list.append([key.name,[i.name for i in value],int(type(key)==Interval)])
    imm_parents = reader.problem.language.immediate_parent
    imm_parent_list = []
    for key,value in imm_parents.items():
        if 'name' not in dir(value):
            imm_parent_list.append([key.name,None,int(type(key)==Interval)])
        else:
            imm_parent_list.append([key.name,value.name,int(type(key)==Interval)])
    return ancestor_list, imm_parent_list



if __name__ == '__main__':
    parse_model('pr-domain.pddl','pr-problem.pddl')

import re
import os
import torch
import json
import yaml
import random
import numpy as np
from pathlib import Path

try:
    from tarski.io import PDDLReader
except:
    raise ImportError("To run experiments on blocksworld, please install tarski "
                      "with `pip install tarski`.")
try:
    from pddl.logic import Predicate, constants, variables
    from pddl.core import Domain, Problem, Action, Requirements
    from pddl.formatter import domain_to_string, problem_to_string
    from pddl import parse_problem as parse_pddl_problem
except:
    raise ImportError("To run experiments on blocksworld, please install pddl "
                      "with `pip install pddl`.")

# helper functions from https://github.com/karthikv792/LLMs-Planning

def instance_to_text_blocksworld(problem, get_plan, data, plan_code="", shuffle=False):
    """Function to make a blocksworld instance into human-readable format
    
    :param get_plan: Flag to return the plan as text as well
    """

    OBJS = data['encoded_objects']

    # ----------- PARSE THE PROBLEM ----------- #
    INIT, GOAL = parse_problem(problem, data, shuffle)

    # ----------- PLAN TO TEXT ----------- #
    PLAN = ""
    plan_file = "sas_plan"
    if get_plan:
        PLAN = "\n"
        if plan_code != "":
            plan = plan_code.split("\n")[:-1]
        else:
            with open(plan_file) as f:
                plan = [line.rstrip() for line in f][:-1]

        for action in plan:
            action = action.strip("(").strip(")")
            act_name, objs = action.split(" ")[0], action.split(" ")[1:]
            objs = [OBJS[obj] for obj in objs]
            PLAN += data['actions'][act_name].format(*objs) + "\n"
        PLAN += "[PLAN END]\n"

    return INIT, GOAL, PLAN

def parse_problem(problem, data, shuffle):
    def get_sorted(init_atoms):
        return sorted(init_atoms, key=lambda x: x.symbol.name+" "+\
                      " ".join([subterm.name for subterm in x.subterms]))

    def parse(init_goal_preds, OBJS):
        TEXT = ""
        predicates = []

        init_goal_preds = list(init_goal_preds)
        for atom in init_goal_preds:
            objs = []
            for subterm in atom.subterms:
                objs.append(OBJS[subterm.name])
            predicates.append(data['predicates'][atom.symbol.name].format(*objs))
        if len(predicates) > 1:
            TEXT += ", ".join(predicates[:-1]) + f" and {predicates[-1]}"
        else:
            TEXT += predicates[0]
        return TEXT
    
    OBJS = data['encoded_objects']

    init_atoms = get_sorted(problem.init.as_atoms())
    goal_preds = get_sorted(problem.goal.subformulas) if hasattr(problem.goal, 'subformulas') else [problem.goal]

    if shuffle:
        random.shuffle(init_atoms)
        random.shuffle(goal_preds)

    # ----------- INIT STATE TO TEXT ----------- #
    INIT = parse(init_atoms, OBJS)

    # ----------- GOAL TO TEXT ----------- #
    GOAL = parse(goal_preds, OBJS)

    return INIT, GOAL

def fill_template(INIT, GOAL, PLAN):
    text = ""
    if INIT != "":
        text += "\n[STATEMENT]\n"
        text += f"As initial conditions I have that, {INIT.strip()}."
    if GOAL != "":
        text += f"\nMy goal is to have that {GOAL}."
    text += f"\n\nMy plan is as follows:\n\n[PLAN]{PLAN}"

    # TODO: Add this replacement to the yml file -- Use "Translations" dict in yml
    text = text.replace("-", " ").replace("ontable", "on the table")
    return text

def read_config(config_file):
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_problem(instance, domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain)
    return reader.parse_instance(instance)

# defined for RAP

def compute_plan(domain, instance, plan_file="sas_plan"):
    fast_downward_path = os.getenv("FAST_DOWNWARD")
    # Remove > /dev/null to see the output of fast-downward
    assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
    cmd = f"{fast_downward_path}/fast-downward.py {domain} {instance} --search \"astar(lmcut())\" > /dev/null 2>&1"
    os.system(cmd)

    if not os.path.exists(plan_file):
        raise Exception("Plan not found. Check PDDL Writer.")

    return Path(plan_file).read_text()
    

def get_intermediate_states(domain_path, instance, config_data, shuffle=False):
    problem_path, plan_code, _ = instance
    plan_path = "temp_plan"
    temp_problem_path = "temp_problem"
    with open(plan_path, "w") as f:
        f.write(plan_code)
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate -v {domain_path} {problem_path} {plan_path}"
    response = os.popen(cmd).read()
    change_str = response.split("-----------------------")[-1]\
        .split("Plan executed successfully")[0]\
        .strip().split("\n\n")
    changes = []
    for c in change_str:
        changes.append(c.split("\n")[1:])
    problem = parse_pddl_problem(problem_path)
    # pddls = []
    states = []
    cur_state = problem.init
    even = True
    for change in changes:
        even = not even
        del_list = [c.replace("Deleting ", "") for c in change if "Deleting" in c]
        add_list = [c.replace("Adding ", "") for c in change if "Adding" in c]
        s = set()
        for i in cur_state:
            if str(i) not in del_list:
                s.add(i)
        for i in add_list:
            s.add(Predicate(* i[1:-1].split(" ")))
        p = Problem(name=problem.name, domain_name=problem.domain_name, requirements=problem.requirements, objects=problem.objects, init=s.copy(), goal=problem.goal)
        with open(temp_problem_path, "w") as f:
            f.write(problem_to_string(p))
        temp_problem = get_problem(temp_problem_path, domain_path)
        if even:
            TEMP_INIT, TEMP_GOAL, TEMP_PLAN = instance_to_text_blocksworld(temp_problem, False, config_data, plan_code="")
            states.append(TEMP_INIT)
        # pddls.append(problem_to_string(p))
        cur_state = s
    return states

def load_blocksworld(config_file, domain_file, data_file=None, data_list=None, return_intermediate=False):
    assert data_file is not None and data_list is None or data_file is None and data_list is not None
    if data_file is not None:
        data_list = json.load(open(data_file, 'r'))
    config_data = read_config(config_file)
    domain_pddl = domain_file
    data = []
    for cur_instance in data_list:
        cur_data = {}
        problem = get_problem(cur_instance[0], domain_pddl)
        gt_plan_code = cur_instance[1]
        # compute_plan(domain_pddl, cur_instance[0], "sas_plan")
        INIT, GOAL, PLAN = instance_to_text_blocksworld(problem, True, config_data, plan_code=gt_plan_code)
        cur_data["init"] = INIT
        cur_data["goal"] = GOAL
        cur_data["plan"] = PLAN
        if return_intermediate:
            states = get_intermediate_states(domain_pddl, cur_instance, config_data)
            cur_data["states"] = states
        # cur_data["icl"] = prompt["icl"]
        # gt_plan = compute_plan(domain_pddl, cur_instance[0])
        # cur_data["gt_plan"] = gt_plan
        cur_data["question"] = fill_template(
            *instance_to_text_blocksworld(problem, False, config_data)) + "\n"
        cur_data["instance_file"] = cur_instance[0]
        data.append(cur_data)
    return data

def get_ordered_objects(object_names, line):
    objs = []
    pos = []
    for obj in object_names:
        if obj in line:
            objs.append(obj)
            pos.append(line.index(obj))

    sorted_zipped_lists = sorted(zip(pos, objs))
    return [el for _, el in sorted_zipped_lists]

def text_to_plan_blocksworld(text, cur_instance_file, config_file, domain_pddl, plan_file, ground_flag=False):

    data = read_config(config_file)
    problem = get_problem(cur_instance_file, domain_pddl)
    action_set = problem.actions
    # ----------- GET DICTIONARIES ----------- #
    LD = data['encoded_objects']  # Letters Dictionary
    BD = {v: k for k, v in LD.items()}  # Blocks Dictionary

    # ----------- GET RAW AND TEXT-FORMATTED ACTIONS AND OBJECTS ----------- #
    actions_params_dict = dict(action_set.items())
    raw_actions = list(action_set.keys())
    text_actions = [x.replace("-", " ") for x in raw_actions]

    text = text.lower().strip()
    for raw_action, text_action in zip(raw_actions, text_actions):
        text = text.replace(text_action, raw_action)

    object_names = [x.lower() for x in LD.values()]

    # ----------- GET PLAN FROM TEXT ----------- #
    plan = ""
    readable_plan = ""
    lines = [line.strip() for line in text.split("\n")]
    for line in lines:
        if '[COST]' in line:
            break
        # Extracting actions
        action_list = [action in line.split() for action in raw_actions]
        if sum(action_list) == 0:
            continue
        # TODO: Handle GPT-3 text that can't be parsed as an action
        action = raw_actions[np.where(action_list)[0][0]]
        # Extracting Objects
        n_objs = len(actions_params_dict[action].parameters.vars())
        objs = get_ordered_objects(object_names, line)
        if len(objs) != n_objs:
            continue
        readable_objs = [obj.replace(' block', '') for obj in objs]
        objs = [BD[x] for x in objs]
        readable_action = "({} {})".format(action, " ".join(readable_objs[:n_objs + 1]))
        if not ground_flag:
            action = "({} {})".format(action, " ".join(objs[:n_objs + 1]))
        else:
            action = "({}_{})".format(action, "_".join(objs[:n_objs + 1]))

        plan += f"{action}\n"
        readable_plan += f"{readable_action}\n"

    print(f"[+]: Saving plan in {plan_file}")
    file = open(plan_file, "wt")
    file.write(plan)
    file.close()

    return plan, readable_plan

def validate_plan(domain, instance, lm_plan_file):
    """Validate the plan using VAL

    :param domain: domain file
    :param instance: instance file
    :param lm_plan_file: plan file (saved earlier)
    """
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {lm_plan_file}"
    response = os.popen(cmd).read()

    print("RESPONSE:::", response)
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')

    if "Plan valid" in response:
        return True, response
    return False, response


def generate_all_actions(state):
    """Generate all possible actions from the current state

    :param state: current state
    """
    return_list = []
    if "hand is empty" in state:
        block = re.findall("the [a-z]{0,10} block is clear", state)
        block_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
        for c in block_color:
            if f"the {c} block is on the table" in state:
                return_list.append(f"pick up the {c} block")
            else:
                c_ = re.search(f"the {c} block" + " is on top of the ([a-z]{0,10}) block", state).group(1)
                return_list.append(f"unstack the {c} block from on top of the {c_} block")
    else:
        c = re.search("is holding the ([a-z]{0,10}) block", state).group(1)
        block = re.findall("the [a-z]{0,10} block is clear", state)
        clear_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
        for c_ in clear_color:
            return_list.append(f"stack the {c} block on top of the {c_} block")
        return_list.append(f"put down the {c} block")
    return return_list


def apply_change(change, state):
    """Apply the predicted change to the state
    
    :param change: predicted change
    :param state: current state
    """
    if "and the " in state and ", and the" not in state:
        state = state.replace("and the ", ", and the ")
    states = state.split(", ")
    states = [s.strip()[4:].strip(".") if s.strip().startswith("and ")\
               else s.strip().strip(".") for s in states]
    changes = change.lower().strip().strip(".").split(", ")
    for c in changes:
        if c.startswith("and "):
            c = c[4:]
        success = 0
        if c.startswith("the hand"):
            old = c.split("was")[1].split("and")[0].strip()
            new = c.split("now")[1].strip()
            for idx in range(len(states)):
                if ("hand is " + old) in states[idx]:
                    states[idx] = states[idx].replace(old, new)
                    success += 1
        else:
            
            colors = re.findall(r"the (\w+) block", c)
            if len(colors) == 0:
                print("Error: zero-colors")
                print(c)

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                
                raise Exception("ERROR")
            color = colors[0]
            if c.startswith(f"the {color} block"):
                subj = f"{color} block"
                if "no longer" in c:
                    old = c.split("no longer")[1].strip()
                    # print("old:", old)
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = ""
                            success += 1
                elif "was" in c and "now" in c:
                    old = c.split("was")[1].split(" and")[0].strip()
                    new = c.split("now")[1].strip()
                    # print("previous:", "{color} block is " + old)
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = states[idx].replace(old, new)
                            success += 1
                elif "now" in c:
                    new = c.split("now")[1].strip()
                    states.append("the " + color + " block is " + new)
                    success += 1
            else:
                print("Error: not recognized")
                print(c)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                raise Exception("ERROR")
        
        if success == 0:
            print("Error: no successful change")
            print(c)
            print(states)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            raise Exception("ERROR")
    states = [s for s in states if s != ""]
    priority_states = []
    for s in states:
        if "have that" in s:
            priority_states.append(0)
        elif "clear" in s:
            priority_states.append(1)
        elif "in the hand" in s:
            priority_states.append(1)
        elif "the hand is" in s:
            priority_states.append(2)
        elif "on top of" in s:
            priority_states.append(3)
        elif "on the table" in s:
            priority_states.append(4)
        else:
            print("Error: unknown state")
            print(s)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            raise Exception("ERROR")
    sorted_states = [x.strip() for _, x in sorted(zip(priority_states, states))]
    sorted_states[-1] = "and " + sorted_states[-1]
    return ", ".join(sorted_states) + "."


def goal_check(goals, blocks_state):
    """Check if the goals are met and return the percentage of goals met

    :param goals: goals
    :param blocks_state: current blocks state
    """
    meetings = [g in blocks_state for g in goals]
    # print("Goals:", goals)
    # print("Goal met:", meetings)
    if sum(meetings) == len(meetings):
        return True, 1.0
    return False, sum(meetings) / len(meetings)

def extract_goals(example, return_raw=False):
    """Extract the goals from the example
    
    :param example: example
    """
    goal_statement = example["question"].split("[STATEMENT]")[-1]\
        .split("My goal is to ")[1].split("My plan is as follows")[0].strip()
    if return_raw:
        return goal_statement
    goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
    return goals

def extract_init_state(example):
    """Extract the initial state from the example
    
    :param example: example
    """
    # print(example)
    init_statement = example["question"].split("[STATEMENT]\nAs initial conditions I have that, ")[1]\
        .split("My goal")[0].strip()
    return init_statement
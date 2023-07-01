import re
import os
import torch

def validate_plan(domain, instance, plan_file):
    """Validate the plan using VAL

    :param domain: domain file
    :param instance: instance file
    :param plan_file: plan file (saved earlier)
    """
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {plan_file}"
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
                return_list.append(f"Pick up the {c} block")
            else:
                c_ = re.search(f"the {c} block" + " is on top of the ([a-z]{0,10}) block", state).group(1)
                return_list.append(f"Unstack the {c} block from on top of the {c_} block")
    else:
        c = re.search("is holding the ([a-z]{0,10}) block", state).group(1)
        block = re.findall("the [a-z]{0,10} block is clear", state)
        clear_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
        for c_ in clear_color:
            return_list.append(f"Stack the {c} block on top of the {c_} block")
        return_list.append(f"Put down the {c} block")
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
                torch.distributed.barrier()
                raise Exception("ERROR")
        if success == 0:
            print("Error: no successful change")
            print(c)
            print(states)
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
    if sum(meetings) == len(meetings):
        return True, 1.0
    return False, sum(meetings) / len(meetings)

def extract_goals(example):
    """Extract the goals from the example
    
    :param example: example
    """
    goal_statement = example["question"].split("[STATEMENT]")[-1]\
        .split("My goal is to ")[1].split("[PLAN]")[0]
    goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
    return goals

def extract_init_state(example):
    """Extract the initial state from the example
    
    :param example: example
    """
    init_statement = example["question"].split("[STATEMENT]\nAs initial conditions ")[1]\
        .split("my goal")[0].strip()
    return init_statement
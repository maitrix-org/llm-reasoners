import random
import os


def get_action_text(action, data):
    pred = action.split('_')
    if 'blocksworld' in data['domain_name']:
        objs = [data["encoded_objects"][j] for j in pred[1:]]
    elif 'logistics' in data['domain_name']:
        # print(pred)
        objs = [data["encoded_objects"][obj[0]].format(*[chr for chr in obj if chr.isdigit()]) for obj in pred[1:]]
    return data['actions'][pred[0]].format(*objs)





# --------------- CHAIN OF THOUGHT (not implemented completely) ----------------- #
def get_state_translation(state, data):
    DATA = data
    STATE = ""
    state_text = []
    for i in sorted(state):
        pred = i.split('_')
        if 'obfuscated' in DATA["domain_name"]:
            objs = [j.replace('o','object_') for j in pred[1:]]
        elif 'blocksworld' in DATA["domain_name"]:
            objs = [DATA["encoded_objects"][j] for j in pred[1:]]
        elif 'logistics' in DATA["domain_name"]:
            objs = [DATA["encoded_objects"][obj[0]].format(*[chr for chr in obj if chr.isdigit()]) for obj in pred[1:]]
        # ADD SPECIFIC TRANSLATION FOR EACH DOMAIN HERE
        try:
            state_text.append(DATA['predicates'][pred[0]].format(*objs))
        except KeyError:
            # print(f"KeyError: {pred[0]}")
            pass
        
    if len(state_text) > 1:
        STATE += ", ".join(state_text[:-1]) + f" and {state_text[-1]}"
    elif len(state_text) == 1:
        STATE += state_text[0]
    

    # STATE += "."
    return STATE

def paraphrase_goal(exec, data):
    exec.complete_plan_execution()
    goal_state, full_state = list(exec.goal_state), list(exec.final_state)
    random.shuffle(goal_state)
    return len(goal_state), get_state_translation(full_state, data)

def generate_plan_cot(planexecutor, data, give_response):
    """
    We need
        i. Initial State
       ii. Plan subset
      iii. Resulting state
    If prompt:
        Give Initial State, Plan and Goal State
    else:
        Give Initial State and Resulting State as Goal State.
    :return:
    """
    initial_state = planexecutor.init_state
    # print(initial_state)
    # planexecutor.random_prefix_execution()
    goal_state = planexecutor.goal_state
    resulting_state = planexecutor.final_state
    DATA = data
    INIT = get_state_translation(initial_state, DATA)
    PLAN = "[PLAN]"
    if give_response:
        """
        Format of Plan:
        1.  Current State: __get current state__
            Action: __get action__
            Reason: __get reason__
            Resulting State: __get resulting state__
        2.  Current State: __get current state__
            Action: __get action__
            Resulting State: __get resulting state__
        ...
        n.  Current State: __get current state__
            Action: __get action__
            Resulting State: __get resulting state__
        Final State: __get final state__
        The goal conditions are satisfied in the final state. Hence, the above plan is a valid plan.       

        """


        plan_text = ""
        start, end = 0, 0
        steps = 1
        state = initial_state
        for index, i in enumerate(planexecutor.plan):
            start = end
            end = index + 1
            plan_text += f"\n{steps}. Current State: {get_state_translation(state, DATA)}\n"
            state = planexecutor.get_final_state(state, planexecutor.plan[start:end])
            preconds = planexecutor.get_action_preconditions(i.upper())
            precondition_text = get_state_translation(preconds, DATA)
            action = get_action_text(i, DATA)
            plan_text += f"   Action: {action}\n"
            plan_text += f"   Reason: The above action is applicable in the current state because its preconditions; {precondition_text}, are satisfied in the current state.\n"

            plan_text += f"   Resulting State: {get_state_translation(state, DATA)}\n"
            steps += 1
        plan_text += f"\nFinal State: {get_state_translation(state, DATA)}\n"
        plan_text += "The goal conditions are satisfied in the final state. Hence, the above plan is valid.\n[PLAN END]\n"
        PLAN += plan_text
    else:
        plan_text = "\n"
        for i in planexecutor.plan[:planexecutor.prefix]:
            action = get_action_text(i, DATA)
            plan_text += action
            plan_text += "\n"
        # PLAN+=plan_text

    GOAL = get_state_translation(goal_state, DATA)
    # goal_text = []
    # if give_response:
    #     for i in goal_state:
    #         pred = i.split('_')
    #         objs = [DATA["encoded_objects"][j] for j in pred[1:]]
    #         goal_text.append(DATA['predicates'][pred[0]].format(*objs))
    # else:
    #     for i in goal_state:
    #         pred = i.split('_')
    #         objs = [DATA["encoded_objects"][j] for j in pred[1:]]
    #         goal_text.append(DATA['predicates'][pred[0]].format(*objs))

    # if len(goal_text) > 1:
    #     GOAL += ", ".join(goal_text[:-1]) + f" and {goal_text[-1]}"
    # else:
    #     GOAL += goal_text[0]

    text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}\nMy goal is to have that {GOAL}.\nMy plan is as follows:\n\n{PLAN}"
    return text, plan_text
def parsed_instance_to_text_blocksworld(initial_state, plan, goal_state, data, action_seq=False, is_grounded=False):
    INIT = get_state_translation(initial_state, data)
    PLAN = ""
    plan_text = "\n"
    for i in plan:
        action = get_action_text(i, data)
        plan_text += action + "\n"
    if not action_seq:
        plan_text += "[PLAN END]\n"
    else:
        plan_text += "[ACTION SEQUENCE END]\n"
    PLAN += plan_text

    GOAL = get_state_translation(goal_state, data)

    return INIT, PLAN, GOAL


def generate_plan_subset(planexecutor, data, give_response):
    """
    We need
        i. Initial State
       ii. Plan subset
      iii. Resulting state
    If prompt:
        Give Initial State, Plan and Goal State
    else:
        Give Initial State and Resulting State as Goal State.
    :return:
    """
    initial_state = planexecutor.init_state
    planexecutor.random_prefix_execution()
    goal_state = planexecutor.goal_state
    resulting_state = planexecutor.final_state
    # Get differene between initial state and resulting state
    
    if give_response:
        INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(initial_state, planexecutor.plan, goal_state, data, is_grounded=planexecutor.is_pr_grounded)
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}.\nMy plan is as follows:\n\n[PLAN]{PLAN} "
        return text, PLAN
    else:
        INIT, _, GOAL = parsed_instance_to_text_blocksworld(initial_state,
                                                            planexecutor.plan[:planexecutor.prefix],
                                                            resulting_state, data, is_grounded=planexecutor.is_pr_grounded)
        PLAN_PREFIX = planexecutor.plan[:planexecutor.prefix]
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}.\nMy plan is as follows:\n\n[PLAN]"
        return text, PLAN_PREFIX


def optimality(planexecutor, data, give_response=True):
    """
    We need
        i. Initial State
        ii. Goal
        iii. Plan
        iv. Cost for plan
    :param exec:
    :param data:
    :param give_response:
    :return:
    """
    initial_state = planexecutor.init_state
    goal_state = planexecutor.goal_state
    plan = planexecutor.plan
    cost = planexecutor.cost
    COST = ""
    INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(initial_state, plan, goal_state, data, is_grounded=planexecutor.is_pr_grounded)
    COST += f"The total time to execute the plan is {cost} minute"
    if cost > 1:
        COST += "s.\n"
    else:
        COST += ".\n"
    if give_response:
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}. I want to minimize the time taken to achieve my goal.\nMy plan is as follows:\n\n[PLAN]{PLAN}{COST}"
    else:
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}. I want to minimize the time taken to achieve my goal.\nMy plan is as follows:\n\n[PLAN] "
    return text, PLAN + COST


def replanning(planexecutor, data, give_response, is_harder=random.choice(([0, 1]))):
    """

    :return:
    """
    if is_harder:
        hard = "Problem was made harder\n"
    else:
        hard = "Problem was made easier\n"

    initial_state = planexecutor.init_state
    goal_state = planexecutor.goal_state
    to_add_or_remove = planexecutor.replanning_domain_specific(is_harder, domain=data['domain_name'])
    # print("PREFIX:", planexecutor.prefix)
    final_action = planexecutor.plan[:planexecutor.prefix][-1]
    new_model = planexecutor.get_new_instance(change_goal=False, change_init=True)
    plan, cost = planexecutor.get_plan('pr-new-domain.pddl', 'pr-new-problem.pddl')
    replanning_state = planexecutor.replanning_init
    if is_harder:
        execution_text = f"During execution, an unexpected event has occurred.\nAfter executing the action \"{get_action_text(final_action, data)}\" in the plan, The following facts unexpectedly became false: {get_state_translation(to_add_or_remove, data)}"
    else:
        execution_text = f"During execution, an unexpected event has occurred.\nAfter executing the action \"{get_action_text(final_action, data)}\" at step {planexecutor.prefix} in the plan, the following facts unexpectedly became true: {get_state_translation(to_add_or_remove['to_add'], data)}\nThe following facts became unexpectedly false: {get_state_translation(to_add_or_remove['to_remove'], data)}"
    INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(initial_state, planexecutor.plan, goal_state, data, is_grounded=planexecutor.is_pr_grounded)
    text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}.\nMy plan is as follows:\n\n[PLAN]{PLAN}\n"
    text += execution_text
    INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(replanning_state, plan, goal_state, data, is_grounded=planexecutor.is_pr_grounded)
    if give_response:
        text += f"\nAfter re-planning from the new state, the plan is as follows:\n[PLAN]{PLAN}"
    else:
        text += f"\nAfter re-planning from the new state, the plan is as follows:\n[PLAN]"
        # text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}\nMy goal is to have that {GOAL}.\nMy plan is as follows:\n\n[PLAN]"
    return text, plan, new_model


def plan_execution(planexecutor, data, give_response):
    """
    We need
        i. Initial State
       ii. Plan subset
      iii. Resulting state
    If prompt:
        Give Initial State, Plan Subset, the resulting state
    else:
        Give Initial State, Plan Subset
    :return:
    """
    initial_state = planexecutor.init_state
    planexecutor.random_prefix_execution()
    plan_prefix = planexecutor.plan[:planexecutor.prefix]
    resulting_state = planexecutor.final_state

    INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(initial_state, plan_prefix, [], data, action_seq=True)
    if give_response:
        FIN = f'[RESULTING STATE]\n{get_state_translation(resulting_state, data)}\n'
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\n I have executed the following action sequence:\n\n[ACTION SEQUENCE]{PLAN}{FIN}"
    else:
        FIN = f'[RESULTING STATE]\n'
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\n I have executed the following action sequence:\n\n[ACTION SEQUENCE]{PLAN}{FIN}"

    return text, list(resulting_state)

def plan_verification_zero_shot(planexecutor, data, llm_plan=None):
    if llm_plan is None:
        example_type = random.choice([-1, 0, 1])
        plan, cost = planexecutor.get_plan(planexecutor.pr_domain, planexecutor.pr_problem)       

        if example_type == -1: #Inexecutable 
            if len(plan)>2:
                to_del = random.choice(range(1, len(plan)-1))
            else:
                to_del = 1
            plan = plan[:to_del]+plan[to_del+1:]
            random.shuffle(plan)
        elif example_type==0: #Unsatisfied goal
            #Pick a prefix of the plan
            prefix= random.choice(range(0, len(plan)-1))
            plan = plan[:prefix]
        else:
            pass
    else:
        plan = llm_plan
        plan = [action.replace('(', '').replace(')', '') for action in plan]
        plan = ["_".join(action.split(' ')) for action in plan]
    
    initial_state = planexecutor.init_state
    goal_state = planexecutor.goal_state    
    INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(initial_state, plan, goal_state, data)
    text = f'\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}. \nMy plan is as follows:\n\n[PLAN]{PLAN}\nVerify whether the above plan is valid. If it is valid, please say "Plan is valid." and nothing else. If it is invalid, please say "Plan is invalid." and then provide feedback on why the plan fails.'
    return text


def plan_verification_zero_shot_val_form(planexecutor, data, llm_plan=None):
    if llm_plan is None:
        example_type = random.choice([-1, 0, 1])
        plan, cost = planexecutor.get_plan(planexecutor.pr_domain, planexecutor.pr_problem)       

        if example_type == -1: #Inexecutable 
            if len(plan)>2:
                to_del = random.choice(range(1, len(plan)-1))
            else:
                to_del = 1
            plan = plan[:to_del]+plan[to_del+1:]
            random.shuffle(plan)
        elif example_type==0: #Unsatisfied goal
            #Pick a prefix of the plan
            prefix= random.choice(range(0, len(plan)-1))
            plan = plan[:prefix]
        else:
            pass
    else:
        plan = llm_plan
        plan = [action.replace('(', '').replace(')', '') for action in plan]
        plan = ["_".join(action.split(' ')) for action in plan]
    
    initial_state = planexecutor.init_state
    goal_state = planexecutor.goal_state    
    INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(initial_state, plan, goal_state, data)
    text = f'\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}. \nMy plan is as follows:\n\n[PLAN]{PLAN}\nVerify whether the above plan is valid. If it is valid, please say "Plan is valid." and nothing else. If it is invalid, please say "Plan is invalid." and then provide feedback on why the plan fails according to the following format. If the plan is inexecutable, provide the first action that is inexecutable and the unmet preconditions in the following format: The following action [action name] has unmet preconditions [list of preconditions]. If the plan is executable but does not satisfy the goal, provide the unmet goal conditions.'
    return text
    

def plan_verification(planexecutor, data, run_val, give_response=False, example_type=None, llm_plan=None):
    '''
    Generates a single plan verification prompt for a single plan. If there is
    no existing plan, a plan can be generated by selecting an example type. If there is
    an existing plan, this should be supplied as the llm plan. Only one of llm plan
    or example type is needed.

    val can be used to get the true result for this plan and this result can be appended
    to the text. If the ground truth plan is not needed (such as when verification is done 
    in back prompting), val can be skipped. If val is run, give_response determines
    whether the result is added to the text. An example of where this is useful
    is when we provide the LLM with an example of a plan and what its result should be.
    If val is not run, give_response is ignored.
    '''
    if llm_plan is None:
        plan, cost = planexecutor.get_plan(planexecutor.pr_domain, planexecutor.pr_problem)       

        if example_type == -1: #Inexecutable 
            if len(plan)>2:
                to_del = random.choice(range(1, len(plan)-1))
            else:
                to_del = 1
            plan = plan[:to_del]+plan[to_del+1:]
            random.shuffle(plan)
        elif example_type==0: #Unsatisfied goal
            
            prefix= random.choice(range(0, len(plan)-1))
            plan = plan[:prefix]
        else:
            pass
        if example_type == 1:
            val_message = "The above plan is valid.\n"
        else:
            with open('sas_plan_ver', 'w') as f:
                if planexecutor.is_pr_grounded:
                    for action in plan:
                        f.write(f'({action})\n')
                else:
                    for action in plan:
                        f.write(f'({" ".join(action.split("_"))})\n')
            if run_val:
                domain = planexecutor.pr_domain
                problem = planexecutor.pr_problem
                val_feedback_dict = get_val_feedback(domain, problem, 'sas_plan_ver')
                val_message = get_validation_message(val_feedback_dict, data)
    else:
        plan = llm_plan
        with open('sas_plan_ver', 'w') as f:
            for action in plan:
                if planexecutor.is_pr_grounded:
                    action = "_".join(action.split(' '))
                f.write(f'{action}\n')
        if run_val:
            domain = planexecutor.pr_domain
            problem = planexecutor.pr_problem
            val_feedback_dict = get_val_feedback(domain, problem, 'sas_plan_ver')
            val_message = get_validation_message(val_feedback_dict, data)
        plan = [action.replace('(', '').replace(')', '') for action in plan]
        plan = ["_".join(action.split(' ')) for action in plan]
    initial_state = planexecutor.init_state
    goal_state = planexecutor.goal_state    
    INIT, PLAN, GOAL = parsed_instance_to_text_blocksworld(initial_state, plan, goal_state, data)
    if run_val and give_response:
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}. \nMy plan is as follows:\n\n[PLAN]{PLAN}\n[VERIFICATION]\n{val_message}"
    else:
        text = f"\n[STATEMENT]\nAs initial conditions I have that, {INIT.strip()}.\nMy goal is to have that {GOAL}. \nMy plan is as follows:\n\n[PLAN]{PLAN}\n[VERIFICATION]"
    return text, val_message if run_val else None
        



def reformat_feedback(feedback):
        unmet_precond = []
        unmet_goal = []
        precond = False
        goal = False
        for line in feedback:
            line = line.strip()
            if 'unsatisfied precondition' in line:
                precond = True
                goal = False
                unmet_precond.append(f'Time step: {line.split(" ")[-1]}\nAction: {line[line.find("("):line.find(")")+1]}\nUnsatisfied Precondition:\n')
                continue
            elif 'goal is not satisfied' in line:
                goal = True
                precond = False
                unmet_goal.append(f'There are unmet goal condition(s):\n')
                continue
            if precond and line:
                is_false = True if 'false' in line else False
                
                if 'Follow each of:' in line:
                    line = line.replace('Follow each of:', 'and')
                elif 'Follow one of:' in line:
                    line = line.replace('Follow one of:', 'or')
                if 'and (Set' in line:
                    line = line.replace("and (Set ", '').replace(' to true)', '').replace(' to false)', '')
                elif '(Set' in line:
                    line = line.replace("(Set ", '').replace(' to true)', '').replace(' to false)', '')
                if is_false:
                    line = f'(not {line})'
                unmet_precond.append(line)
            elif goal and line:
                if 'Follow each of:' in line:
                    line = line.replace('Follow each of:', 'and')
                elif 'Follow one of:' in line:
                    line = line.replace('Follow one of:', 'or')
                if 'and (Set' in line:
                    line = line.replace("and (Set ", '').replace(' to true)', '').replace(' to false)', '')
                elif '(Set' in line:
                    line = line.replace("(Set ", '').replace(' to true)', '').replace(' to false)', '')
                unmet_goal.append(line)
        return unmet_precond, unmet_goal

def get_val_feedback(domain_file, instance_file, plan_file):
    val = os.environ.get('VAL')
    cmd = f'{val}/validate -v {domain_file} {instance_file} {plan_file}'
    response = os.popen(cmd).read()
    plan_valid = 'Plan valid' in response
    feedback = []
    repair = False
    for line in response.split('\n'):
        if 'Plan Repair' in line:
            repair = True
            continue
        if 'Failed plans' in line:
            repair = False
        if repair and line:
            feedback.append(line)
    print(feedback)
    unmet_precond, unmet_goal = reformat_feedback(feedback)
    feedback_dict = {
        'validation_info': {'is_valid_plan': plan_valid},
        'validation_message': '\n'.join(unmet_goal) if unmet_goal else '\n'.join(unmet_precond),
        'unmet_info': {'unmet_precond': unmet_precond, 'unmet_goal': unmet_goal}
    }
    return feedback_dict
        

def get_validation_message(val_message, data):
    unmet_precond, unmet_goal = val_message['unmet_info']['unmet_precond'], val_message['unmet_info']['unmet_goal']
    
    error_message = "The above plan is invalid."

    if unmet_goal:
        is_joint = "and" in unmet_goal[1]
        first_predicate = 2 if is_joint else 1
        last_predicate = len(unmet_goal) - 1 if is_joint else 2
        predicates = unmet_goal[first_predicate:last_predicate]
        error_message += " These are the unmet goal conditions:\n" if len(predicates) > 1 else \
            " This is the unmet goal condition:\n"
    elif unmet_precond:
        timestep = unmet_precond[0].split("\n")[0].split(" ")[-1]
        action = unmet_precond[0].split("(")[1].split(")")[0].replace(" ", "_")
        is_joint = "(and" in unmet_precond[1]
        first_predicate = 2 if is_joint else 1
        last_predicate = len(unmet_precond) - 1 if is_joint else 2
        predicates = unmet_precond[first_predicate:last_predicate]
        error_message += f" The following action at step {timestep} has unmet preconditions:\n" \
            if len(predicates) > 1 else \
            f"The following action at step {timestep} has an unmet precondition:\n"
        error_message += get_action_text(action, data) + "\n"
        error_message += "The unmet preconditions are:\n" \
            if len(predicates) > 1 else \
            "The unmet precondition is:\n"
        
    else:
        return None
    error_message += get_state_translation(map(lambda pddl: pddl.strip("()").replace(" ", "_"), predicates), data)
    
    return error_message
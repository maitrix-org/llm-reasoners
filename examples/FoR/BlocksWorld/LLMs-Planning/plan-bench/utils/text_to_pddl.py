import numpy as np 

def get_ordered_objects(object_names, line):
    objs = []
    pos = []
    for obj in object_names:
        if obj in line:
            objs.append(obj)
            pos.append(line.index(obj))

    sorted_zipped_lists = sorted(zip(pos, objs))
    return [el for _, el in sorted_zipped_lists]
def text_to_plan(text, action_set, plan_file, data, cot=False, ground_flag=False):
    if cot:
        plan = []
        for line in text.split("\n"):
            if line.strip() == "":
                continue
            if 'Action:' in line:
                plan.append(line.split(":")[1].strip())
        text = "\n".join(plan)
    if 'obfuscated' in data['domain_name']:
        return text_to_plan_obfuscated(text, action_set, plan_file, data, ground_flag)
    elif data['domain_name']=='logistics':
        return text_to_plan_logistics(text, action_set, plan_file, data, ground_flag)
    elif 'blocksworld' in data['domain_name']:
        return text_to_plan_blocksworld(text, action_set, plan_file, data, ground_flag)
    elif 'depots' in data['domain_name']:
        return text_to_plan_depots(text, action_set, plan_file, data, ground_flag)
    # ADD SPECIFIC TRANSLATION FOR EACH DOMAIN HERE




def has_digit(string):
    return any(char.isdigit() for char in string)
def text_to_plan_logistics(text, action_set, plan_file, data, ground_flag=False):
    raw_actions = [i.split('-')[0].lower() for i in list(action_set.keys())]
    # ----------- GET PLAN FROM TEXT ----------- #
#     load package_0 into airplane_0 at location_0_0
# load package_1 into airplane_1 at location_1_0
# fly airplane_0 from location_0_0 to location_1_0
# fly airplane_1 from location_1_0 to location_0_0
# unload package_0 from airplane_0 at location_1_0
# unload package_1 from airplane_1 at location_0_0
    plan = ""
    readable_plan = ""
    lines = [line.strip().lower() for line in text.split("\n")]
    for line in lines:
        if not line:
            continue
        if '[COST]' in line:
            break
        
        if line[0].isdigit() and line[1]=='.':
            line = line[2:]
            line = line.replace(".", "")
        elif line[0].isdigit() and line[1].isdigit() and line[2]=='.':
            line = line[3:]
            line = line.replace(".", "")

        objs = [i[0]+'-'.join(i.split('_')[1:]) for i in line.split() if has_digit(i)]
        
        
        if line.split()[0] in raw_actions:
            action = line.split()[0]
            if 'load' in action or 'unload' in action:  
                to_check = objs[1]
            else:
                to_check = objs[0]
            if 'a' in to_check:
                action+='-airplane'
            elif 't' in to_check:
                action+='-truck'
            else:
                print(line, objs)
                raise ValueError
            if action=='drive-truck' and len(objs)==3:
                objs.append("c"+[i for i in objs[1] if i.isdigit()][0])


            readable_action = "({} {})".format(action, " ".join(objs))
            if not ground_flag:
                action = "({} {})".format(action, " ".join(objs))
            else:
                action = "({}_{})".format(action, "_".join(objs))
            plan += f"{action}\n"
            readable_plan += f"{readable_action}\n"
    # print(f"[+]: Saving plan in {plan_file}")
    file = open(plan_file, "wt")
    file.write(plan)
    file.close()
    return plan, readable_plan



def text_to_plan_depots(text, action_set, plan_file, data, ground_flag=False):
    raw_actions = [i.lower() for i in list(action_set.keys())]
    plan = ""
    readable_plan = ""
    lines = [line.strip().lower() for line in text.split("\n")]
    for line in lines:
        if not line:
            continue
        if '[COST]' in line:
            break
        
        line = line.lstrip("0123456789").replace(".","")
        print(line)
        #line = re.sub("^[0-9]+.","",line)
        #KAYQ must the stripping of periods only happen when it's part of a list?

        objs = [i for i in line.split() if has_digit(i)]
        
        found_flag = False
        for x in raw_actions:
            if x in line:
                action = x
                found_flag = True
                continue
        if not found_flag:
            continue

        readable_action = "({} {})".format(action, " ".join(objs))
        if not ground_flag:
            action = "({} {})".format(action, " ".join(objs))
        else:
            action = "({}_{})".format(action, "_".join(objs))
        plan += f"{action}\n"
        readable_plan += f"{readable_action}\n"
    # print(f"[+]: Saving plan in {plan_file}")
    file = open(plan_file, "wt")
    file.write(plan)
    file.close()
    return plan, readable_plan



def text_to_plan_obfuscated(text, action_set, plan_file, data, ground_flag=False):
    """
    Converts obfuscated domain's plan in plain text to PDDL plan
    ASSUMPTIONS:
        (1) Actions in the text we have them on the domain file
        (2) We know the object names
        (3) Objects order is given by the sentence

    :param text: Obfuscated text to convert
    :param action_set: Set of possible actions
    :param plan_file: File to store PDDL plan
    """


    # object_names = [x.lower() for x in LD.values()]
    raw_actions = data['actions'].keys()
    # ----------- GET PLAN FROM TEXT ----------- #
    plan = ""
    readable_plan = ""
    lines = [line.strip() for line in text.split("\n")]
    for line in lines:
        if '[COST]' in line:
            break
        # Extracting actions
        if line.strip() == "":
            continue
        action_list = [action in line.split() for action in raw_actions]
        object_list = [obj.strip() for obj in line.split('object_') if obj.strip().isdigit()==True]
        if sum(action_list) == 0:
            continue
        if len(object_list) == 0:
            continue
        action = raw_actions[np.where(action_list)[0][0]]
        # Extracting Objects
        n_objs = data['actions']['action'].count('{}')        
        objs = ['o'+o for o in object_list]
        if len(objs) != n_objs:
            continue
        readable_objs = [obj.replace('o', 'object_') for obj in objs]
        readable_action = "({} {})".format(action, " ".join(readable_objs[:n_objs + 1]))
        if not ground_flag:
            action = "({} {})".format(action, " ".join(objs[:n_objs + 1]))
        else:
            action = "({}_{})".format(action, "_".join(objs[:n_objs + 1]))

        plan += f"{action}\n"
        readable_plan += f"{readable_action}\n"
    # print(f"[+]: Saving plan in {plan_file}")
    file = open(plan_file, "wt")
    file.write(plan)
    file.close()

    return plan, readable_plan

def text_to_plan_blocksworld(text, action_set, plan_file, data, ground_flag=False):
    """
    Converts blocksworld plan in plain text to PDDL plan
    ASSUMPTIONS:
        (1) Actions in the text we have them on the domain file
        (2) We know the object names
        (3) Objects order is given by the sentence

    :param text: Blocksworld text to convert
    :param action_set: Set of possible actions
    :param plan_file: File to store PDDL plan
    """

    # ----------- GET DICTIONARIES ----------- #
    LD = data['encoded_objects']  # Letters Dictionary
    BD = {v: k for k, v in LD.items()}  # Blocks Dictionary
    AD = {}  # Action Dictionary
    for k, v in data['actions'].items():
        word = v.split(' ')[0]
        if word in k:
            AD[k] = k.replace("-", " ")
        else:
            AD[k] = word

    # ----------- GET RAW AND TEXT-FORMATTED ACTIONS AND OBJECTS ----------- #
    actions_params_dict = dict(action_set.items())
    raw_actions = [i.lower() for i in list(action_set.keys())]
    text_actions = [AD[x] for x in raw_actions]

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
    # print(f"[+]: Saving plan in {plan_file}")
    file = open(plan_file, "wt")
    file.write(plan)
    file.close()

    return plan, readable_plan


def text_to_state(text, data):
    text_preds = text.replace(' and ',',').split(",")
    if 'mystery' in data['domain_name']:
        return text_to_state_mystery(text_preds, data)
    elif ' obfuscated' in data['domain_name']:
        return text_to_state_obfuscated(text_preds, data)
    elif 'logistics'in data['domain_name']:
        return text_to_state_logistics(text_preds, data)
    elif 'blocksworld' in data['domain_name']:
        return text_to_state_blocksworld(text_preds, data)
    elif 'depots' in data['domain_name']:
        return text_to_state_depots(text_preds, data)
    # ADD SPECIFIC TRANSLATION FOR EACH DOMAIN HERE

def text_to_state_obfuscated(preds, data):
    pddl_state = []
    for pred in preds:
        pred = pred.strip()
        if pred == '':
            continue
        if ' not ' in pred:
            continue
        pddl_pred = ''
        pddl_map = ''
        for map in data['predicates']:
            if map in pred:
                pddl_pred = map
                pddl_map = data['predicates'][map]
                break
        if pddl_pred == '':
            continue
        objs = []
        for obj in pred.split('object_'):
            if obj.strip.isdigit():
                objs.append('o'+obj.strip())
        pddl_pred += '_'+ '_'.join(objs)
        pddl_state.append(pddl_pred)
    return pddl_state

def text_to_state_mystery(preds, data):
    pddl_state = []
    for pred in preds:
        pred = pred.strip()
        if pred == '':
            continue
        if ' not ' in pred:
            continue
        pddl_pred = ''
        pddl_map = ''
        for map in data['predicates']:
            if map in pred:
                pddl_pred = map
                pddl_map = data['predicates'][map]
                break
        if pddl_pred == '':
            continue
        objs = []
        for obj in pred.split(pddl_map):
            for block in data['encoded_objects']:
                if data['encoded_objects'][block] in obj:
                    objs.append(block)
                    break
        pddl_pred += '_'+ '_'.join(objs)
        pddl_state.append(pddl_pred)
    return pddl_state

def text_to_state_blocksworld(preds, data):
    blocks  = dict([(v.replace(' block',''),k) for k,v in data['encoded_objects'].items()])
    pddl_state = []
    for pred in preds:
        pred = pred.strip()
        if pred == '':
            continue
        if ' not ' in pred:
            continue
        pddl_pred = ''
        pddl_map = ''
        for map in data['predicate_mapping']:
            if data['predicate_mapping'][map] in pred:
                pddl_pred = map
                pddl_map = data['predicate_mapping'][map]
                break
        if pddl_pred == '':
            continue
        objs = []
        for obj in pred.split(pddl_map):
            for block in blocks:
                if block in obj:
                    objs.append(block)
                    break
        param_count = data['predicates'][pddl_pred].count('{}')
        for obj in objs[:param_count]:
            pddl_pred += '_' + blocks[obj]
        pddl_state.append(pddl_pred)
    return pddl_state





def text_to_state_logistics(preds, data):
    pddl_state = []
    for pred in preds:
        pred = pred.strip()
        if pred == '':
            continue
        if ' not ' in pred:
            continue
        if ' is at ' in pred:
            objs = [i for i in pred.split(' is at ') if len(i)>0]
            pddl_pred = 'at_' + '_'.join(objs)
        elif ' is in ' in pred:
            objs = [i for i in pred.split(' is in ') if len(i)>0]
            pddl_pred = 'in_' + '_'.join(objs)
        else:
            continue
        pddl_state.append(pddl_pred)
    return pddl_state




def text_to_state_depots(preds, data):
    pddl_state = []
    for pred in preds:
        pred = pred.strip()
        if pred == '':
            continue
        if ' not ' in pred:
            continue
        if ' is at ' in pred:
            objs = [i for i in pred.split(' is at ') if len(i)>0]
            pddl_pred = 'at_' + '_'.join(objs)
        elif ' is in ' in pred:
            objs = [i for i in pred.split(' is in ') if len(i)>0]
            pddl_pred = 'in_' + '_'.join(objs)
        elif ' is on ' in pred:
            objs = [i for i in pred.split(' is on ') if len(i)>0]
            pddl_pred = 'on_' + '_'.join(objs)
        else:
            continue
        pddl_state.append(pddl_pred)
    return pddl_state
 
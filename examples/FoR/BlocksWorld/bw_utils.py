import re
import torch
from util import *
import warnings
from transformers.trainer_pt_utils import LabelSmoother
import copy
import random
import openai
warnings.filterwarnings("ignore")

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def generate_all_actions(state):
    return_list = []
    if "hand is empty" in state:
        block = re.findall("the [a-z]{0,10} block is clear", state)
        block_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
        
        for c in block_color:

            if f"the {c} block is on the table" in state:
                return_list.append(f"Pick up the {c} block")
            else:
                try:
                    c_ = re.search(f"the {c} block" + " is on top of the ([a-z]{0,10}) block", state).group(1)
                except Exception as e:

                    print("c: ", c)
                    print("state: ", state)

                    import time
                    time.sleep(1) 
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

    if "and the " in state and ", and the" not in state:
        state = state.replace("and the ", ", and the ")
    states = state.split(", ")
    states = [s.strip()[4:].strip(".") if s.strip().startswith("and ") else s.strip().strip(".") for s in states]
    

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

                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = ""
                            success += 1
                elif "was" in c and "now" in c:
                    old = c.split("was")[1].split(" and")[0].strip()
                    new = c.split("now")[1].strip()
                    
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
            return "Infeasible"

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

def query_LM(worldmodel, tokenizer, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.7):
    temperature = temperature if do_sample else 0
    all_results = []
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    
    results = worldmodel.generate(input_ids, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    input_ids_list = input_ids.squeeze().tolist()
    input_len = len(input_ids_list)

    results = tokenizer.decode(results[0][input_len:], skip_special_tokens=False)
    last_newline_position = results.find('\n')
    results = results[:last_newline_position] if last_newline_position != -1 else results
    all_results.append(prompt + results)
    return all_results

def preprocess(tokenizer, state, actions):
    IGNORE_TOKEN_ID = LabelSmoother.ignore_index
    input_ids = torch.tensor(tokenizer.encode(state, return_tensors="pt")).squeeze()
    action_ids = [torch.tensor(tokenizer.encode(action, return_tensors="pt")).squeeze() for action in actions]

    action_positions = []
    
    for action_id in action_ids:
        if action_id.dim() == 0 or input_ids.shape[-1] - action_id.shape[-1] <= 0:
            continue
        for i in range(input_ids.shape[-1] - action_id.shape[-1], -1, -1):
            if torch.equal(input_ids[i:i + action_id.shape[-1]], action_id):
                action_positions.append((i, i + action_id.shape[-1] - 1))
                break
    
    target = [IGNORE_TOKEN_ID] * input_ids.shape[-1]

    for action_position in action_positions:
        target[action_position[0]: action_position[1]] = input_ids[action_position[0]: action_position[1]]

    target = torch.tensor(target)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    return dict(
        input_ids=input_ids.unsqueeze(0),
        labels=target.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0)
    )

def sample_prompt(init_prompt,
                shuffle_prompt=True,
                num_shot=4):
    
    if shuffle_prompt:
        examples = random.sample(init_prompt["example_pool"], num_shot)
    else:
        examples = init_prompt["example_pool"][:num_shot]

    icl = get_icl(init_prompt, examples)
    
    prompt = copy.deepcopy(init_prompt)
    prompt["icl"] = icl
    prompt["icl_list"] = [icl]
    examples = copy.deepcopy(examples)
    for i in range(5):
        new_examples = []
        for example in examples:
            if len(example["states"]) > 1:
                new_examples.append({
                    "init": example["states"][0],
                    "goal": example["goal"],
                    "plan": "\n" + "\n".join(example["plan"].split("\n")[3:]),
                    "states": example["states"][1:]
                })
            else:
                new_examples.append(example)
        examples = copy.deepcopy(new_examples)
        icl = get_icl(init_prompt, examples)
        prompt["icl_list"].append(icl)
    return prompt

def get_icl(init_prompt, examples):
    icl = init_prompt["intro"] + \
        "\n".join([
            "[STATEMENT]\nAs initial conditions I have that, " + \
            example["init"] + \
            ".\nMy goal is to have that " +\
            example["goal"] + \
            ".\n\nMy plan is as follows:\n\n[PLAN]" + \
            example["plan"]
            for example in examples
        ])
    icl += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
    return icl
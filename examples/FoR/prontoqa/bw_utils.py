import re
import torch
from util import *
import warnings
from transformers.trainer_pt_utils import LabelSmoother
import copy
import random
import openai
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
warnings.filterwarnings("ignore")

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_quant_type="nf4",
    llm_int8_threshold=6.0,
    bnb_4bit_use_double_quant=True,
)
world_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                        trust_remote_code=True,
                                        device_map="auto",
                                        torch_dtype=torch.bfloat16,
                                        quantization_config=bnb_config)
world_model.eval()

print(world_model.device)
world_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", add_bos_token=False, padding_side='left')
world_tokenizer.pad_token = world_tokenizer.eos_token
"""

def generate_all_actions_old(state):
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

def query_LM(world_model, world_tokenizer, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.7, max_new_tokens=50):
    temperature = temperature if do_sample else 0
    all_results = []
    input_ids = world_tokenizer.encode(prompt, return_tensors='pt').to(world_model.device)
    
    results = world_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=world_tokenizer.eos_token_id)

    input_ids_list = input_ids.squeeze().tolist()
    # 获取 input_ids 的长度
    input_len = len(input_ids_list)

    results = world_tokenizer.decode(results[0][input_len:], skip_special_tokens=False)
    last_newline_position = results.find('\n')

    results = results[:last_newline_position] if last_newline_position != -1 else results
    all_results.append(prompt + results)
    return all_results



def query_LM_no(worldmodel, tokenizer, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.7):
    temperature = temperature if do_sample else 0
    all_results = []
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    results = worldmodel.generate(input_ids, max_new_tokens=200, do_sample=True, top_k=10, eos_token_id=eos_token_id)
    results = tokenizer.decode(results[0], skip_special_tokens=False)
    last_newline_position = results.rfind('\n')
    
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",  # 选择适当的引擎
        
        prompt=prompt,
        max_tokens=2000,
        temperature=0.05,
        stop=['\n'])
    results = response.choices[0].text.strip()
    all_results.append(results)
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

def generate_trajectories(initial_state,
                          goal,
                          prompts,
                          model,
                          tokenizer,
                          max_steps,
                          temperature,
                          eos_token_id, 
                          agent=None,
                          
                          ):
    """
    return: trajs, probability of each action in the trajs, log rewards of the trajs, log rewards of (state, action)
    """
    base_prompt = "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nAfter being given an initial state and an action, give the new state after performing the action.\n"

    state = base_prompt + prompts["goal_prefix"] + goal.strip() + "\n" + prompts["state_prefix"].format(0) + " " + initial_state.strip() + "\n"
   

    actions = []

    
    for step in range(max_steps):
        
        
        state += prompts["action_prefix"].format(step + 1) + " "
        input_ids = tokenizer.encode(state, return_tensors='pt').to(model.device)

        agent_output = model.generate(input_ids, max_length=input_ids.shape[-1] + 20, do_sample=True, top_k=10, eos_token_id=eos_token_id)
        decoded_output = tokenizer.decode(agent_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        newline_position = decoded_output.find('\n')

        action = decoded_output[:newline_position] if newline_position != -1 else decoded_output
       
        state += action

        actions.append(action)

        last_state = re.search(f'.*{re.escape(prompts["state_prefix"].format(step))}(.*)', state)[1]
        last_action = re.search(f'.*{re.escape(prompts["action_prefix"].format(step+1))}(.*)', state)[1]
        if "Pick" in last_action or "Pick".lower() in last_action: 
            world_update_prompt = prompts["world_update_pickup"].format(last_state, last_action)
        elif "Unstack" in last_action or "Unstack".lower() in last_action:
            world_update_prompt = prompts["world_update_unstack"].format(last_state, last_action)
        elif "Put" in last_action or  "Put".lower() in last_action:
            world_update_prompt = prompts["world_update_putdown"].format(last_state, last_action)
        elif "Stack" in last_action or "Stack".lower() in last_action: 
            world_update_prompt = prompts["world_update_stack"].format(last_state, last_action)
        lora_to_base(model) # transfer to World Model
        world_output = query_LM(model, tokenizer, world_update_prompt, do_sample=False, num_return_sequences=1,
                                    eos_token_id=eos_token_id)[0]

        
        world_change = world_output.split("[CHANGE]")[-1]

        last_state = state.split(f"[STATE {step}]")[-1].split(f"[ACTION {step+1}]")[0]

        
        new_state = apply_change(world_change, last_state)
        
        if new_state == "Infeasible":
            # 
            goal_statement = state.split("[GOAL]")[-1].split("[STATE 0]")[0]
            goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
            meetings = [g in last_state for g in goals]
            if sum(meetings) == len(meetings):
                r1 = 100
            else:
                r1 = sum(meetings) / len(meetings) + 0.5

            sample = preprocess(tokenizer, state, actions)
            
            r1 = torch.tensor(r1)
            return state, actions, torch.log(r1), sample
    
        state += "\n" + prompts["state_prefix"].format(step+1) + " " + new_state + "\n"

        
        

        base_to_lora(model)

       

    goal_statement = state.split("[GOAL]")[-1].split("[STATE 0]")[0]
    goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
    meetings = [g in new_state for g in goals]
    if sum(meetings) == len(meetings):
        r1 = 100
    else:
        r1 = sum(meetings) / len(meetings) + 0.5
    r1 = torch.tensor(r1)
    sample = preprocess(tokenizer, state, actions)

    return state, actions, torch.log(r1), sample



def sample_prompt(init_prompt,
    shuffle_prompt=True,
                num_shot=2):
    
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
import torch
import torch
import numpy as np
from util import *
from abc import ABC, abstractmethod
import torch

import os

def get_full_reward(gt, actions, sum="sum"):
    reward = torch.zeros(len(actions), dtype=torch.float32, device="cuda:0")
    num_feasible_actions = int((len(gt)-1)/2)
    j = 0
    for i in range(num_feasible_actions):
        gt_action = gt[2*i+1][0]
        if (i < len(actions)) and (actions[i].lower() in gt_action.lower()):
            reward[i] += 100
            j += 1
        else:
            break
  
    if sum=="sum":
        ret  = torch.sum(reward, dtype=torch.float32) 
    elif sum == "avg":
        ret  = torch.mean(reward, dtype=torch.float32) 
    if ret.item() == 0:
        ret += 0.0001
    return ret

def eval_tf(last_state, query, answer):

    if set(last_state.lower().replace("not", "").split()) == set(query[len("True or false: "):].lower().replace("not", "").split()):
        # finish:
        if answer == "True":
            if query[len("True or false: "):].lower() == last_state.lower():
                return True
        else:
            if "not" in set(last_state.lower().split()) - set(query[len("True or false: "):].lower().split()) or "not" in set(query[len("True or false: "):].lower().split()) - set(last_state.lower().split()):
                return True

    return False

def is_finish(last_state, query):
    if ("(" in last_state.strip())  or ("No conclusion can be drawn from these facts".lower() in last_state.lower()) or (query[len("True or false: "):].lower().split()[0] not in last_state.lower()):
        print("FINISH!!!!:\n", last_state, query)
        return True
    return set(last_state.strip().lower().replace("not", "").split()) == set(query[len("True or false: "):].lower().replace("not", "").split())


def compute_ppl_line(model, tokenizer, line):
    line = line.strip()

    line_ = tokenizer.encode(line)
    line_t = torch.tensor(line_, dtype=torch.long).cuda()
    loss = model(input_ids=line_t, labels=line_t).loss
    loss = loss.detach().clone().data.cpu().numpy()
    ppl = np.exp(loss)
    return ppl

def lora_to_base(model):
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()
    
def base_to_lora(model):
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()

def tb_loss_old(log_pf, log_r, logz, log_bf=None):
    print("log_pf: ", log_pf)
    print("log_r: ", log_r)
    print("logz: ", logz)
    print("log_bf:", log_bf)
    if log_bf != None:
        loss = (log_pf + logz - log_r - log_bf).pow(2).mean()
    else:
        loss = (log_pf + logz - log_r).pow(2).mean()
    return loss

def tb_loss(log_pf, log_r, logz, log_bf=None, logpartition=True):
    print("log_pf: ", log_pf)
    print("log_r: ", log_r)
    print("logz: ", logz)
    if logpartition:
        if log_bf != None:
            scores = log_pf - log_r - log_bf
            loss = (scores - scores.mean()) ** 2 
        else:
            scores = log_pf - log_r
            loss = (scores - scores.mean()) ** 2 
    else:
        if log_bf != None:
            loss = (log_pf + logz - log_r - log_bf) ** 2
        else:
            loss = (log_pf + logz - log_r) ** 2
    return loss.mean()

class QueryLM(ABC):
    @abstractmethod
    def query_LM(self, prompt, **gen_kwargs):
        pass

    @abstractmethod
    def query_next_token(self, prompt: list[str]):
        pass

class QueryLlama(QueryLM):
    def __init__(self, llamamodel, max_response_length, log_file) -> None:
        self.llamamodel = llamamodel
        self.tokenizer = self.llamamodel.tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = llamamodel.model.params.max_batch_size
        self.yes_no = self.tokenizer.encode('Yes No', bos=False, eos=False)

    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        temperature = temperature if do_sample else 0
        all_results = []
        for start in range(0, num_return_sequences, self.max_batch_size):
            end = min(start + self.max_batch_size, num_return_sequences)
            results = self.llamamodel.generate([prompt] * (end - start), max_gen_len=self.max_response_length, temperature=temperature, eos_token_id=eos_token_id)
            all_results.extend(results)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("="*50+"\n")
                f.write(prompt + "\n")
                for result in all_results:
                    f.write("-"*50+"\n")
                    f.write(result.replace(prompt, "") + "\n")
        return all_results

    @torch.no_grad()
    def query_next_token(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        ret = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
            tokens = torch.tensor([tokens]).cuda().long()
            output, h = self.llamamodel.model.forward(tokens, start_pos=0)
            ret.append(output)
        outputs = torch.cat(ret, dim=0)
        filtered = outputs[:, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist


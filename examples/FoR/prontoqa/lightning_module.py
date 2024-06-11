import random
import sys
import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers.trainer_pt_utils import LabelSmoother
from util import lora_to_base, base_to_lora
from bw_utils import *
from typing import Tuple, Union, Optional
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import csv
import re
from torch.distributions import Categorical

def extract_last_state(content):
    match = re.findall(r'\[STATE \w+\]', content)
    if match:
        last_state_index = content.rfind(match[-1])
        if last_state_index != -1:
            last_state_content = content[last_state_index + len(match[-1]):]
            end_index = last_state_content.find('\n')
            if end_index != -1:
                return last_state_content[:end_index].strip()


class BlocksWorldGFNTask(LightningModule):
    def __init__(
        self,
        args,
        model,
        logZ,
        tokenizer,
        replay_buffer,
        train_data=None,
        val_data=None,
        test_data=None
    ):
        super().__init__()

        self.args = args
        self.logZ = logZ
        self.model = model
        if args.use_lora:
            base_to_lora(self.model)

        self.tokenizer = tokenizer
        self.reward = None
        self.replay_buffer = replay_buffer
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.n_samples = args.n_samples 

        self.lr = args.lr
        self.logZ_lr = args.logZ_lr

        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)

        self.get_reward_temp_at_step = lambda step: self.args.reward_temp_start + (
           self.args.reward_temp_end - self.args.reward_temp_start
        ) * min(1, step / self.args.reward_temp_horizon)

        self.ignore_token_id = LabelSmoother.ignore_index

        self.reward_temperature = self.args.reward_temp_start
        self.pf_temperature = self.args.pf_temp_start

        self.epsilon = self.args.epsilon_start

        transition_path = f"transitions/{self.args.step}/transition.json"

        self.wrong_transitions = {}

        self.ls_wrong_transitions = {}

        if os.path.exists(transition_path):
            with open(transition_path, 'r') as f:
                self.transitions = json.load(f)
        else:
            self.transitions = {}
        


    def forward(self, problem, pf_temperature=1.0):

        ACTIONS, QUERY, PLAN, GT = problem
        GT = GT[0]
        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]

        (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
        ) = self.generate_trajectories_v2(
            query = QUERY,
            allowed_actions = ACTIONS,
            gt = GT,
            plan = PLAN,
            temperature=pf_temperature,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0]
                            )

        return generated_text, actions, states, sample, reward

    def training_step(self, problem, batch_idx):
        self.wrong_transitions = {}

        ACTIONS, QUERY, PLAN, GT = problem

        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]
        GT = GT[0]
      
        ########################## Compute the reward for ground-truth trajectory ##########################

        LOG_R = []
        LOG_PF = []
        LOG_BF = []
        # Exploitation: Reuse the samples in the buffer

        if (
            random.random() < self.args.use_buffer_prob
            and self.replay_buffer.sample(self.n_samples, QUERY)[0] is not None
        ):
            # Using a sample from the reward buffer
            (log_reward_list,
            state_list,
            sample_list
            ) = self.replay_buffer.sample(
                self.n_samples, QUERY
            )

            for state, sample in zip(state_list, sample_list):
                (actions, states) = eval(state)
                log_pf, log_bf = self.forward_prob(QUERY, ACTIONS, actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)
            LOG_R.extend(log_reward_list)
        
        else:
            best_actions = None
            best_states = None
            best_reward = -9999
            for _ in range(self.n_samples):
                if np.random.rand() < self.args.pf_temp_prob:
                    pf_temp = self.pf_temperature
                else:
                    pf_temp = 0.7
                generated_text, actions, states, sample, reward = self.forward(
                    problem, pf_temp
                )
                
                ll_reward = self.get_ll_reward_rule_hard(actions, states, QUERY)

                if self.args.sum_avg == "sum":
                    log_r = torch.log(self.args.ll_weight * ll_reward.sum())
                    LOG_R.append(log_r)
        
                elif self.args.sum_avg == "avg":
                    log_r = torch.log(self.args.ll_weight * ll_reward.mean())
                    LOG_R.append(log_r)
                
                print("generated ll: \n",  ll_reward)
                print("trajectory query: \n",  QUERY)
                print("trajectory states: \n",  states)
                print("trajectory actions: \n",  actions)
                """
                log_r = torch.log(reward)
                LOG_R.append(log_r)
                print("reward: \n",  log_r)
                print("trajectory query: \n",  QUERY)
                print("trajectory states: \n",  states)
                print("trajectory actions: \n",  actions)
                """
                generated_text = (actions, states)
                self.replay_buffer.add(QUERY, str(generated_text), sample, log_r)

                log_pf, log_bf = self.forward_prob(QUERY, ACTIONS, actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)

                if log_r > best_reward:
                    best_actions  = actions
                    best_states = states
                    best_reward = log_r

            self.ls_wrong_transitions = {}
            for _ in range(6):
                _, actions, states, reward, _ = self.local_search(
                        query = QUERY,
                        allowed_actions = ACTIONS,
                        #gt = GT,
                        gt_plan= PLAN,
                        past_actions = best_actions,
                        eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0])
                
                ll_reward = self.get_ll_reward_rule_hard(actions, states, QUERY)
                print("generated ll_ls: \n",  ll_reward)
                print("trajectory query_ls: \n",  QUERY)
                print("trajectory states_ls: \n",  states)
                print("trajectory actions_ls: \n",  actions)

                if self.args.sum_avg == "sum":
                    log_r = torch.log(self.args.ll_weight * ll_reward.sum())
        
                elif self.args.sum_avg == "avg":
                    log_r = torch.log(self.args.ll_weight * ll_reward.mean())
                """
                log_r = torch.log(reward)
                print("reward_ls: \n",  log_r)
                print("trajectory query_ls: \n",  QUERY)
                print("trajectory states_ls: \n",  states)
                print("trajectory actions_ls: \n",  actions)
                """
                # if log_r is larger, then we accept it
                if log_r > best_reward:
                    LOG_R.append(log_r)
                    generated_text = (actions, states)
                    self.replay_buffer.add(QUERY, str(generated_text), sample, log_r)
                    log_pf, log_bf = self.forward_prob(QUERY, ACTIONS, actions, states)
                    LOG_PF.append(log_pf)
                    LOG_BF.append(log_bf)

            self.ls_wrong_transitions = {}

        LOG_PF = torch.stack(LOG_PF).to(self.model.device)
        LOG_BF = torch.stack(LOG_BF).to(self.model.device)
        LOG_R = torch.stack(LOG_R).to(self.model.device)
        
        LOG_R = LOG_R * (1 / self.reward_temperature)

        base_to_lora(self.model)
    
        self.wrong_transitions = {}
        # Get the Trajectory balance loss
    
        loss = tb_loss(
            log_pf=LOG_PF,
            log_r=LOG_R,
            logz=self.logZ,
            log_bf=None
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "train/logR",
            LOG_R.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )

        return loss

    @torch.no_grad()
    def test_step(self, problem, batch_idx):
        # pass

        base_to_lora(self.model)   
        self.model.eval()           

        ACTIONS, QUERY, PLAN, GT = problem

        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]
        GT = GT[0]

        total_success = 0
        total_proof_success = 0
        success_text = []
        (
        generated_text, 
        actions, 
        states,
        reward, 
        sample
        ) = self.generate_trajectories_v2(
            query = QUERY,
            allowed_actions = ACTIONS,
            gt = GT,
            plan = PLAN,
            temperature=0.5,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
            mode="test",
            argmax=True
        )
        if eval_tf(states[-1], QUERY, GT):
            total_success += 1
            actions_joined = '\n'.join(actions)
            if actions_joined not in success_text:
                success_text.append((QUERY, actions_joined))

        last_3_plans = [PLAN[-5][0], PLAN[-3][0],PLAN[-1][0]]

        if "Finish" not in actions[-1]:
            last_3_actions = actions[-3:]
        else:
            last_3_actions = actions[-4:-1]

        if last_3_actions == last_3_plans:
            total_proof_success += 1
        for _ in range(32):

            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories_v2(
                query = QUERY,
                allowed_actions = ACTIONS,
                gt = GT,
                plan = PLAN,
                temperature=0.5,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            if eval_tf(states[-1], QUERY, GT):
                total_success += 1
                actions_joined = '\n'.join(actions)
                if actions_joined not in success_text:
                    success_text.append((QUERY, actions_joined))

            last_3_plans = [PLAN[-5][0], PLAN[-3][0],PLAN[-1][0]]

            if "Finish" not in actions[-1]:
                last_3_actions = actions[-3:]
            else:
                last_3_actions = actions[-4:-1]

            if last_3_actions == last_3_plans:
                total_proof_success += 1

        with open(self.args.test_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(success_text)

        if total_success > 0:
            success = 1
        else:
            success = 0

        if total_proof_success > 0:
            psuccess = 1
        else:
            psuccess = 0

        self.log(
            "test/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "test/n_solutsion",
            len(success_text),
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )

        self.log(
            "test/psuccess",
            psuccess,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "test/n_psolutsion",
            total_proof_success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        

    @torch.no_grad()
    def validation_step(self, problem, batch_idx):
        # pass
        self.wrong_transitions = {}
        base_to_lora(self.model)    
        self.model.eval()       

        ACTIONS, QUERY, PLAN, GT = problem

        ACTIONS = ACTIONS[0]
        QUERY = QUERY[0]
        GT = GT[0]

        total_success = 0
        total_proof_success = 0
        success_text = []

        #argmax

        (
        generated_text, 
        actions, 
        states,
        reward, 
        sample
        ) = self.generate_trajectories_v2(
            query = QUERY,
            allowed_actions = ACTIONS,
            gt = GT,
            plan = PLAN,
            temperature=0.5,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
            mode="test",
            argmax=True
        )

        if eval_tf(states[-1], QUERY, GT):
            total_success += 1
            s = "success"
        else:
            s = "fail"
        actions_joined = '==>'.join(actions)
        states_joined = '==>>'.join(states)

        #if actions_joined not in success_text:
        last_3_plans = [PLAN[-6][0], PLAN[-4][0],PLAN[-2][0]]
        if "Finish" not in actions[-1]:
            last_3_actions = actions[-3:]
        else:
            last_3_actions = actions[-4:-1]

        if last_3_actions == last_3_plans:
            total_proof_success += 1
            ps = "proof_success"
        else:
            ps = "proof_fail"

        success_text.append((s, ps, QUERY, GT, actions_joined, states_joined))

        for _ in range(32):

            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories_v2(
                query = QUERY,
                allowed_actions = ACTIONS,
                gt = GT,
                plan = PLAN,
                temperature=0.5,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            if eval_tf(states[-1], QUERY, GT):
                total_success += 1
                s = "success"
            else:
                s = "fail"
            actions_joined = '==>'.join(actions)
            states_joined = '==>>'.join(states)

            #if actions_joined not in success_text:
            last_3_plans = [PLAN[-6][0], PLAN[-4][0],PLAN[-2][0]]
            if "Finish" not in actions[-1]:
                last_3_actions = actions[-3:]
            else:
                last_3_actions = actions[-4:-1]

            if last_3_actions == last_3_plans:
                total_proof_success += 1
                ps = "proof_success"
            else:
                ps = "proof_fail"

            success_text.append((s, ps, QUERY, GT, actions_joined, states_joined))

        with open(self.args.valid_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(success_text)

        if total_success > 0:
            success = 1
        else:
            success = 0

        if total_proof_success > 0:
            psuccess = 1
        else:
            psuccess = 0
        
        self.wrong_transitions = {}

        self.log(
            "val/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "val/n_solutsion",
            total_success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )

        self.log(
            "val/psuccess",
            psuccess,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "val/n_psolutsion",
            total_proof_success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        
    
    def on_train_batch_start(self, problem, batch_idx):
        pass

    def on_train_epoch_start(self):
        # Log scheduled quantities
        current_epoch = self.trainer.current_epoch
        if (current_epoch + 1) % 6 == 0:
            self.pf_temperature = self.args.pf_temp_start - (self.args.pf_temp_start - self.args.pf_temp_end) / (self.args.epochs // 6)

        if current_epoch < self.args.epochs // 2:
            self.epsilon = self.args.epsilon_start - (self.args.epsilon_start - self.args.epsilon_end) / (self.args.epochs // 2)
        
        if current_epoch < self.args.epochs // 2:
            self.reward_temperature = self.args.reward_temp_start + current_epoch * (self.args.reward_temp_end - self.args.reward_temp_start) / (self.args.epochs // 2)
        # self.reward_temperature = random.uniform(self.args.reward_temp_start, self.args.reward_temp_end)
        
        # self.epsilon = 0
        self.log("scheduled/R_temperature", self.reward_temperature, sync_dist=True)

    def configure_optimizers(self):
        if self.args.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            optimizer = bnb.optim.PagedAdamW8bit([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5),
                "monitor": "metric_to_track",
                "frequency": 10,
            }
            }
        else:
            return torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])

    def local_search(self,
                    query,
                    allowed_actions,
                    gt_plan,
                    past_actions,
                    eos_token_id,
                    max_steps=10,
                    mode="train",
                          ):
        """
        return: trajs, probability of each action in the trajs, log rewards of the trajs, log rewards of (state, action)
        """
        print("local search starts!!!")
        allowed_actions = allowed_actions.split(". ")
        print("allowed_actions:")
        print(allowed_actions)
        initial_state = allowed_actions[-1]
        allowed_actions = [a+"." for a in allowed_actions[:-1]]

        last_state = initial_state
        print("last_state:\n", last_state)

        actions = []
        finish = False
        step = 0
        states = []
        while not finish and (step <= max(len(gt_plan)+1, max_steps)) and len(allowed_actions) > 0:
            if step < len(past_actions)-1:
                action = past_actions[step]
            else:
                allowed_actions_ = [act for act in allowed_actions if act not in actions]
                if len(allowed_actions_) != 0:
                    action = random.choice(allowed_actions_)
                else:
                    action = random.choice(allowed_actions)

            allowed_actions.remove(action)
            
            if last_state in self.transitions and action in self.transitions[last_state]:
                new_state = self.transitions[last_state][action]
            else:
                with open("data/prompt/state_transit_examples_long.json", "r") as f:
                    dic = json.load(f)
                    world_update_prompt = dic["input"] + dic["facts_format"].format(last_state, action) + dic["next_claim_prefix"] + " "

                lora_to_base(self.model)
                while True:
                    try:
                        world_output = query_LM(self.model, self.tokenizer, world_update_prompt, do_sample=False, num_return_sequences=1,
                                            eos_token_id=eos_token_id)[0][len(world_update_prompt):].strip()
                        new_state = world_output.split("\nClaim")[0].strip()
                        break
                    except Exception as e:
                        print(e)
                        print("An error occurred: Query LM fails, line 721")
                        import time
                        time.sleep(1)

                print("new_state222:\n", last_state)
                print("action222:\n", action)
                print("new_state222:\n", new_state)
                if last_state not in self.transitions:
                    self.transitions[last_state] = {
                        action: new_state
                    }
                elif action not in self.transitions[last_state]:
                    self.transitions[last_state][action] = new_state
                
            finish = is_finish(new_state, query)
            if mode=="train":
                if (not finish) and ((step* 2 + 1) < len(gt_plan)) and (not action in gt_plan[step* 2 + 1][0]):
                    if last_state not in self.ls_wrong_transitions:
                        print("pass here!", action, step)
                        self.ls_wrong_transitions[last_state] = [action]
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    elif action not in self.ls_wrong_transitions[last_state] and (action not in [e[0] for e in gt_plan]):
                        print("pass here222!", action, step)
                        self.ls_wrong_transitions[last_state].append(action)
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    else:
                        print("known wrong pass here222!", action, step)

                else:
                    states.append(last_state)
                    actions.append(action)
                    step += 1
                    last_state = new_state
            else:
                if (not finish) and ((step* 2 + 1) < len(gt_plan)) and (not action in gt_plan[step* 2 + 1][0]):
                    finish = True
                states.append(last_state)
                actions.append(action)
                step += 1
                last_state = new_state

        states.append(last_state)

        r1 = get_full_reward(gt_plan, actions, self.args.sum_avg)

        return None, actions, states, r1, None


    def generate_trajectories_v2(self,
                            query,
                            allowed_actions,
                            gt,
                            plan,
                            temperature,
                            eos_token_id, 
                            max_steps=10,
                            argmax=False,
                            mode="train",
                          ):
        """
        return: trajs, probability of each action in the trajs, log rewards of the trajs, log rewards of (state, action)
        """
        allowed_actions = allowed_actions.split(". ")
        print("allowed_actions:")
        print(allowed_actions)
        initial_state = allowed_actions[-1]
        allowed_actions = [a+"." for a in allowed_actions[:-1]]
        print("query:")
        print(query)
        print("allowed_actions:")
        print(allowed_actions)
        print("gt:")
        print(gt)
        print("plan:\n", plan)
    
        last_state = initial_state
        print("last_state:\n", last_state)

        actions = []
        finish = False
        step = 0
        states = []
        while not finish and (step <= max(len(plan)+1, max_steps)) and len(allowed_actions) > 0:
            base_to_lora(self.model)
            self.model.eval()
            if np.random.rand() < self.epsilon and mode == "train":
                action = random.choice(allowed_actions)
            else:
                with open("data/prompt/next_step_1shot.json", "r") as f:
                    dic = json.load(f)
                    inputs = dic["input"] + dic["facts_format"].format(" ".join(allowed_actions)) + dic["target_format"].format(query) + dic["claim_format"].format(last_state) + dic["next_step_prefix"] + " "
                input_ids = self.tokenizer.encode(inputs, return_tensors='pt').to(self.device)
                
                prefix_output = self.model(input_ids[:, :-1], use_cache=True)
                prefix_past = prefix_output.past_key_values

                action_logits = []
                for a in allowed_actions:
                    action_ids = self.tokenizer.encode(a, add_special_tokens=False,return_tensors='pt').to(self.device)
                    input_ids_with_action = torch.cat([input_ids[:, -1:], action_ids], dim=-1)
                    outputs = self.model(input_ids_with_action, past_key_values=prefix_past, use_cache=True)
                    logits = outputs.logits  
                    total_log_prob = torch.zeros(1).to("cuda:0")
                    for i in range(1, input_ids_with_action.shape[-1]):
                        probs = torch.softmax(logits[:, i - 1, :], dim=-1)
                        for j in range(1):
                            total_log_prob[j] += torch.log(probs[j, input_ids_with_action[j, i]])

                    num_tokens = input_ids_with_action.shape[-1] - 1
                    avg_log_prob = total_log_prob / num_tokens
                    action_logits.append(avg_log_prob)


                action_logits = torch.stack(action_logits) / temperature
                

                action_logits = action_logits.to(torch.float32)
                probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))
                print("probabilities shape\n", probabilities.shape)
                idx = probabilities.argmax()
                print("last_state:\n", last_state)
                print("action space:\n", allowed_actions)
                print("action_idx:\n",idx)
                if not argmax:
                    dist = Categorical(probs=probabilities.t())
                    idx = dist.sample()

                action = allowed_actions[idx]

            allowed_actions.remove(action)

            if last_state in self.transitions and action in self.transitions[last_state]:
                new_state = self.transitions[last_state][action]
            else:
                with open("data/prompt/state_transit_examples_long.json", "r") as f:
                    dic = json.load(f)
                    world_update_prompt = dic["input"] + dic["facts_format"].format(last_state, action) + dic["next_claim_prefix"] + " "

                lora_to_base(self.model)
                while True:
                    try:
                        world_output = query_LM(self.model, self.tokenizer, world_update_prompt, do_sample=False, num_return_sequences=1,
                                            eos_token_id=eos_token_id)[0][len(world_update_prompt):].strip()
                        new_state = world_output.split("\nClaim")[0].strip()
                        break
                    except Exception as e:
                        print(e)
                        print("An error occurred: Query LM fails, line 721")
                        import time
                        time.sleep(1)

                print("new_state222:\n", last_state)
                print("action222:\n", action)
                print("new_state222:\n", new_state)
                if last_state not in self.transitions:
                    self.transitions[last_state] = {
                        action: new_state
                    }
                elif action not in self.transitions[last_state]:
                    self.transitions[last_state][action] = new_state
                
            finish = is_finish(new_state, query)
            if mode=="train":
                if (not finish) and ((step* 2 + 1) < len(plan)) and (not action in plan[step* 2 + 1][0]):
                    if last_state not in self.wrong_transitions:
                        print("pass here!", action, step)
                        self.wrong_transitions[last_state] = [action]
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    elif action not in self.wrong_transitions[last_state] and (action not in [e[0] for e in plan]):
                        print("pass here222!", action, step)
                        self.wrong_transitions[last_state].append(action)
                        states.append(last_state)
                        actions.append(action)
                        last_state = new_state
                        finish = True
                    else:
                        print("known wrong pass here222!", action, step)

                else:
                    states.append(last_state)
                    actions.append(action)
                    step += 1
                    last_state = new_state
            else:
                if (not finish) and ((step* 2 + 1) < len(plan)) and (not action in plan[step* 2 + 1][0]):
                    finish = True
                states.append(last_state)
                actions.append(action)
                step += 1
                last_state = new_state

        states.append(last_state)


        r1 = get_full_reward(plan, actions, self.args.sum_avg)

        return None, actions, states, r1, None

    def get_ll_reward_rule_hard(self, actions, states, goal):
        reward = [0] * len(states)

        for step_idx, (state, action) in enumerate(zip(states, actions)):
            intuition = 0.00001
            if step_idx == 0 or reward[step_idx - 1] != 0.00001:
                if step_idx < len(actions) - 1:
                    next_state = states[step_idx+1]
                    if state.replace(".", "").split(" ")[-1].replace("s", "").lower() in action.replace("s", "").lower():
                        if self.args.sum_avg=="sum":
                            intuition += 20
                        else:
                            intuition += 100
                else:  
                    if state.replace(".", "").split(" ")[-1].replace("s", "").lower() in goal.replace("s", "").lower():
                        if self.args.sum_avg=="sum":
                            intuition += 20
                        else:
                            intuition += 100

            reward[step_idx] = intuition

        return torch.tensor(reward).to(self.device)

    def get_ll_reward_rule(self, actions, states, goal):
        reward = []
        for step_idx, (state, action) in enumerate(zip(states, actions)):
            intuition = 0.0001
            if step_idx < len(actions) - 1:
                next_state = states[step_idx+1]
                if state.replace(".", "").split(" ")[-1].replace("s", "").replace("y", "").lower() in action.replace("s", "").lower():
                    if self.args.sum_avg=="sum":
                        intuition += 1
                    else:
                        intuition += 10

            else:  
                if state.replace(".", "").split(" ")[-1].replace("s", "").lower() in goal.replace("s", "").lower():
                    if self.args.sum_avg=="sum":
                        intuition += 1
                    else:
                        intuition += 10

            reward.append(intuition)

        return torch.tensor(reward).to(self.device)

    @torch.no_grad()
    def get_next_token_logits(self,
                              prompt,
                              candidates):
        # Normalize the prompt to always be a list
        if isinstance(prompt, str):
            prompt = [prompt]

        cand_tokens = []
        for candidate in candidates:
            token = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(token) != 1:
                warnings.warn(f'Candidate "{candidate}" corresponds to {len(token)} tokens instead of 1.')
            cand_tokens.append(token)
            #print(token)
        #print(cand_tokens)

        # Encode prompts
        prompts_tokens = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompt]
        max_prompt_size = max(len(t) for t in prompts_tokens)
        bsz = len(prompts_tokens)

        # Ensure the batch size does not exceed the maximum allowed by the model
        params = self.model.config
        assert bsz <= params.max_position_embeddings, f"Batch size {bsz} exceeds max allowed {params.max_position_embeddings}."

        # Create a tensor for tokens, handling padding
        tokens = torch.full((bsz, max_prompt_size), self.tokenizer.pad_token_id, dtype=torch.long, device='cuda:0')
        for i, t in enumerate(prompts_tokens):
            tokens[i, :len(t)] = torch.tensor(t, device="cuda:0", dtype=torch.long)

        # Generate a matching attention mask
        attention_mask = tokens != self.tokenizer.pad_token_id

        # Obtain logits from the model
        with torch.no_grad():
            outputs = self.model(input_ids=tokens, attention_mask=attention_mask)
            logits = outputs.logits  # Adjust depending on your model's specific output

        # Extract logits for candidate tokens
        # Convert cand_tokens to a PyTorch tensor
        cand_token_tensor = torch.tensor(cand_tokens, dtype=torch.long, device="cuda:0").squeeze()

        # Check if any index is out of bounds
        if (cand_token_tensor >= logits.shape[2]).any():
            print("Index out of bounds detected!")
            return
            # Optionally, adjust or filter out-of-bounds indices here
        else:
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            tokens_prob = probabilities[0, cand_token_tensor]

        #logits = torch.Tensor(logits)
        normalized_prob = tokens_prob / tokens_prob.sum()
        print("return:\n", normalized_prob)
            
        return normalized_prob

    def find_best_match(self, string_list, target_string):
        # Tokenize the target string into a set of words
        target_words = set(target_string.replace(".", "").lower().split())
        
        # Function to count common words
        def count_common_words(entry):
            entry_words = set(entry.replace(".", "").lower().split())
            return len(target_words.intersection(entry_words))
        
        # Apply the count function to each entry and find the index of the max value
        common_counts = [count_common_words(entry) for entry in string_list]
        best_index = common_counts.index(max(common_counts))
        
        return best_index

    def forward_prob(self, query, allowed_actions, actions, states):
        if self.args.use_lora:
            base_to_lora(self.model)
        
        allowed_actions = allowed_actions.split(". ")
        allowed_actions = allowed_actions[:-1]
        allowed_actions = [a+"." for a in allowed_actions]

        print("forward_prob_actions!!!:\n", actions)

        initial_state = states[0]

        last_state = initial_state
        log_pf = []
        log_bf = []


        with open("data/prompt/next_step_1shot.json", "r") as f:
            dic = json.load(f)
            inputs_template = dic["input"] + dic["facts_format"].format(" ".join(allowed_actions))
       

        for step in range(len(actions)):

            inputs = inputs_template + dic["target_format"].format(query) + dic["claim_format"].format(last_state) + dic["next_step_prefix"] + " "

            input_ids = self.tokenizer.encode(inputs, return_tensors='pt').to("cuda:0")
            action = actions[step]
            bsz = len(allowed_actions)  

            action_texts = [ac for ac in allowed_actions]
            action_ids = [self.tokenizer.encode(a, add_special_tokens=False, return_tensors='pt').to("cuda:0") for a in action_texts]
            max_length = max(len(aid[0]) for aid in action_ids)
            padded_action_ids = [torch.cat([aid, torch.full((1, max_length - len(aid[0])), self.tokenizer.pad_token_id, device=self.device)], dim=-1) for aid in action_ids]
            batch_input_ids_with_actions = torch.cat([torch.cat([input_ids, pid], dim=-1) for pid in padded_action_ids], dim=0)
            batch_outputs = self.model(batch_input_ids_with_actions, use_cache=True)
            batch_logits = batch_outputs.logits
            # calculate the probability
            total_log_prob = torch.zeros(bsz).cuda()
            for i in range(input_ids.shape[-1], batch_input_ids_with_actions.shape[-1]):
                probs = torch.softmax(batch_logits[:, i - 1, :], dim=-1)
                for j in range(bsz):
                    if batch_input_ids_with_actions[j, i] != self.tokenizer.pad_token_id:
                        total_log_prob[j] += torch.log(probs[j, batch_input_ids_with_actions[j, i]])
            action_logits = total_log_prob

            # 计算概率分布
            action_logits = action_logits.to(torch.float32)
            probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))

            try:
                idx = allowed_actions.index(action)
            except:
                print("execute find best match:\n", allowed_actions, action)
                idx = self.find_best_match(allowed_actions, action)

            log_pf.append(torch.log(probabilities[idx]))
            
            last_state = states[step+1]
            
            pb = torch.tensor(1 / len(allowed_actions))
            log_bf.append(torch.log(pb))
        return torch.stack(log_pf).sum(), torch.stack(log_bf).sum()

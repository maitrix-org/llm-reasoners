import random
import sys
import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers.trainer_pt_utils import LabelSmoother
from util import lora_to_base, base_to_lora
from bw_utils import *
import yaml
import json
import bitsandbytes as bnb
import csv
import re
import pickle
import os
from transformers import AutoTokenizer, BitsAndBytesConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical
from collections import defaultdict
sys.path.append("gpt-plan-benchmark/gpt_plan_test")

def add_time(text):
    def add_step_statement(text, start_text, end_text, insertion):

        start = text.rfind(start_text)
        if start == -1:
            return text 
        end = text.rfind(end_text)
        if end == -1:
            return text  
        new_text = text[:end] + insertion + text[end:]
        return new_text
        
    insert1 = "STATE <step>: "
    text = text  
    new_text = add_step_statement(text, "[STATEMENT]", "As initial conditions", insert1)
    insert2 = "ACTION <step>: "
    new_text = add_step_statement(new_text, "[PLAN]", "<action>", insert2)
    return new_text

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
    ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.logZ = logZ
        self.model = model

        if args.use_lora:
            base_to_lora(self.model)

        self.tokenizer = tokenizer
        self.reward = None
        self.prompts = json.load(open("data/blocksworld/my_mcts_prompts_update.json", 'r'))
        self.replay_buffer = replay_buffer
        self.train_data = train_data
        self.val_data = val_data
        self.n_samples = args.n_samples # 2 for step 4
        with open('data/blocksworld/bw_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.domain_pddl = f'gpt-plan-benchmark/gpt_plan_test/instances/{self.config["domain_file"]}'

        self.lr = args.lr
        self.logZ_lr = args.logZ_lr
        self.epsilon = self.args.epsilon_start
        self.get_lr_at_step = lambda step: min(step / 20 * self.lr, self.lr)

        self.ignore_token_id = LabelSmoother.ignore_index

        self.reward_temperature = self.args.reward_temp_start
        self.pf_temperature = self.args.pf_temp_start
        self.use_buffer_prob = self.args.use_buffer_prob
        with open(f"./prompts/pool_prompt_v2_step_{args.step}.json") as f:
            self.init_prompt = json.load(f)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            bnb_4bit_use_double_quant=True,
        )
        self.world_tokenizer = AutoTokenizer.from_pretrained(args.world_model, add_bos_token=False, padding_side='left')
        self.world_tokenizer.pad_token = self.world_tokenizer.eos_token

        transition_path = f"/transitions/{args.step}/transition.pkl"

        if os.path.exists(transition_path):
            with open(transition_path, 'rb') as f:
                self.transitions = pickle.load(f)
        else:
            self.transitions = {}
        self.ll_reward_dict = {}

        self.traj = defaultdict(int)

        
    def forward(self, problem, pf_temp):
        INIT, GOAL, PLAN = problem
        GOAL = GOAL[0]
        INIT = INIT[0]

        (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
        ) = self.generate_trajectories(
            initial_state = INIT,
            goal = f'have that {GOAL}.',
            max_steps = self.args.step,
            eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
            pf_temp = pf_temp
        )
        return generated_text, actions, states, sample, reward

    def training_step(self, problem, batch_idx):
        INIT, GOAL, PLAN = problem
        GOAL = GOAL[0]
        INIT = INIT[0]
        initial_state = f'I have that, {INIT}.'
        goal = f'My goal is to have that {GOAL}.'
        actions = PLAN
        ########################## Compute the reward for ground-truth trajectory ##########################

        LOG_R = []
        LOG_PF = []
        LOG_BF = []
        # Exploitation: Reuse the samples in the buffer

        if (
            random.random() < self.use_buffer_prob
            and self.replay_buffer.sample(self.n_samples, GOAL + INIT)[0] is not None
        ):
            # Using a sample from the reward buffer
            (log_reward_list,
            state_list,
            sample_list
            ) = self.replay_buffer.sample(
                self.n_samples, GOAL + INIT
            )

            for state, sample in zip(state_list, sample_list):
                (actions, states) = eval(state)
                log_pf, log_bf = self.forward_prob(f"have that {GOAL}.", actions, states)
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
                    pf_temp = 1.0
                generated_text, actions, states, sample, reward = self.forward(
                    problem, pf_temp
                )

                if self.args.ll_weight == 0:
                    ll_reward = [1 for _ in range(self.args.step)]
                    ll_reward = torch.tensor(ll_reward).to(self.device)
                    ll_weight = 1
                else:
                    ll_reward = self.get_ll_reward(actions, states, f"have that {GOAL}.")
                    ll_reward = -1 / ll_reward
                    ll_weight = self.args.ll_weight

                LOG_R.append(torch.log(reward + ll_weight * ll_reward.sum()))

                generated_text = (actions, states)
                self.replay_buffer.add(GOAL + INIT, str(generated_text), sample, torch.log(reward + ll_weight * ll_reward.sum()))
                log_pf, log_bf = self.forward_prob(f"have that {GOAL}.", actions, states)
                LOG_PF.append(log_pf)
                LOG_BF.append(log_bf)

                actions_joined = '\n'.join(actions)
                self.traj[actions_joined] += 1

                if torch.log(reward + ll_weight * ll_reward.sum()) > best_reward:
                    best_actions  = actions
                    best_states = states
                    best_reward = torch.log(reward + ll_weight * ll_reward.sum())

                # conduct local search
            for _ in range(16):
                _, actions, states, reward, _ = self.local_search(initial_state = f'I have that, {INIT}.',
                    goal = f'My goal is to have that {GOAL}.',
                    max_steps = self.args.step,
                    plan=best_actions, 
                    states=best_states,
                    eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                    pf_temp = pf_temp)

                if self.args.ll_weight == 0:
                    ll_reward = [1 for _ in range(self.args.step)]
                    ll_reward = torch.tensor(ll_reward).to(self.device)
                    ll_weight = 1
                else:
                    ll_reward = self.get_ll_reward(actions, states, f"have that {GOAL}.")
                    ll_reward = -1 / ll_reward
                    ll_weight = self.args.ll_weight

                log_reward = torch.log(reward + ll_weight * ll_reward.sum())

                # if log_reward is larger, then we accept it

                if log_reward > best_reward:
                    LOG_R.append(torch.log(reward + ll_weight * ll_reward.sum()))
                    generated_text = (actions, states)
                    self.replay_buffer.add(GOAL + INIT, str(generated_text), sample, torch.log(reward + ll_weight * ll_reward.sum()))
                    log_pf, log_bf = self.forward_prob(f"have that {GOAL}.", actions, states)
                    LOG_PF.append(log_pf)
                    LOG_BF.append(log_bf)
            

        # Obtain the log_pf and log_reward

        LOG_PF = torch.stack(LOG_PF).to(self.model.device)
        LOG_R = torch.stack(LOG_R).to(self.model.device)
        LOG_BF = torch.stack(LOG_BF).to(self.model.device)
        LOG_R_temperd = LOG_R * self.reward_temperature
        if self.args.use_lora:
            base_to_lora(self.model)
    
        # Get the Trajectory balance loss
    
        loss = tb_loss(
            log_pf=LOG_PF,
            log_r=LOG_R_temperd,
            logz=self.logZ,
            log_bf=None,
            logpartition=True
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
        

    def test_step(self, problem, batch_idx):
        # pass
        if self.args.use_lora:
            base_to_lora(self.model)    # 确保转换成lora
        self.model.eval()           # 必须用eval

        INIT, GOAL, PLAN = problem
        GOAL = GOAL[0]
        INIT = INIT[0]
        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(40):
            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories(
                initial_state = INIT,
                goal = f'have that {GOAL}.',
                max_steps = self.args.step,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            goal_statement = f"My goal is to have that {GOAL}."
            goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
            meetings = [g in states[-1] for g in goals]
            if sum(meetings) == len(meetings):
                total_success += 1
                
                actions_joined = '\n'.join(actions)
                if (GOAL, INIT, actions_joined) not in success_text:
                    total_solution += 1
                    success_text.append((GOAL, INIT, actions_joined))

        if total_success > 0:
            success = 1
        else:
            success = 0

        self.log(
            "test/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "test/n_solutsion",
            total_solution,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )

    def validation_step(self, problem, batch_idx):
        if self.args.use_lora:
            base_to_lora(self.model)    # 确保转换成lora
        self.model.eval()           # 必须用eval

        INIT, GOAL, PLAN = problem
        GOAL = GOAL[0]
        INIT = INIT[0]

        total_success = 0
        total_solution = 0
        success_text = []

        for _ in range(20):

            (
            generated_text, 
            actions, 
            states,
            reward, 
            sample
            ) = self.generate_trajectories(
                initial_state = INIT,
                goal = f'have that {GOAL}.',
                max_steps = self.args.step,
                eos_token_id=self.tokenizer.encode('\n', add_special_tokens=False)[0],
                mode="test"
            )

            goal_statement = f"My goal is to have that {GOAL}."
            goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
            meetings = [g in states[-1] for g in goals]
            
            if sum(meetings) == len(meetings):
                total_success += 1
                actions_joined = '\n'.join(actions)
                if (GOAL, INIT, actions_joined) not in success_text:
                    total_solution += 1
                    success_text.append((GOAL, INIT, actions_joined))

        if total_success > 0:
            success = 1
        else:
            success = 0

        self.log(
            "val/success",
            success,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.batch_size
        )
        self.log(
            "val/n_solutsion",
            total_solution,
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
        
        if current_epoch < self.args.epochs // 2:
            self.use_buffer_prob  = self.args.p_buffer_start + current_epoch * (self.args.p_buffer_end - self.args.p_buffer_start) / (self.args.epochs // 2)
        
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
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
            }
        else:
            return torch.optim.AdamW([{'params': self.model.parameters(), 'lr': self.lr},
                                    {'params': [self.logZ,], 'lr': self.logZ_lr}])

    def generate_trajectories(self,
                            initial_state,
                            goal,
                            max_steps,
                            eos_token_id,
                            pf_temp=1.0,
                            mode="train",
                          ):
        """
        return: trajs, probability of each action in the trajs, log rewards of the trajs, log rewards of (state, action)
        """
        if self.args.use_lora:
            base_to_lora(self.model)
        self.model.eval()
        prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=1)
        last_state = initial_state
        actions = []
        states  = []
        for step in range(max_steps):
            icl_template = prompt["icl_list"][step // 2]
            icl_template = add_time(icl_template)
            previous_action = ""
            current_state = last_state
            allowed_actions = generate_all_actions(last_state)
            allowed_actions_ = [act for act in allowed_actions if act.lower() not in actions]

            if len(allowed_actions_) != 0:

            # epsilon greedy
                if np.random.rand() < self.epsilon and mode == "train":
                    action = random.choice(allowed_actions_)
                    action = action.lower()
                else:
                    inputs = icl_template.replace("<init_state>", current_state.lstrip())\
                        .replace("<goals>", goal).replace("<action>", previous_action.lstrip()).replace("<step>", str(step).strip()).strip()
                    input_ids = self.tokenizer.encode(inputs.lstrip() + "\n", return_tensors='pt').to(self.device)
                    
                    prefix_output = self.model(input_ids[:, :-1], use_cache=True)
                    prefix_past = prefix_output.past_key_values

                    action_logits = []
                    for ac in allowed_actions_:
                        a = ac.lower()
                        action_ids = self.tokenizer.encode(a, add_special_tokens=False,return_tensors='pt').to(self.device)
                        input_ids_with_action = torch.cat([input_ids[:, -1:], action_ids], dim=-1)
                        outputs = self.model(input_ids_with_action, past_key_values=prefix_past, use_cache=True)
                        logits = outputs.logits  # 获取对应于 action_ids 的 logits
                        total_log_prob = torch.zeros(1).cuda()
                        for i in range(1, input_ids_with_action.shape[-1]):
                            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
                            for j in range(1):
                                total_log_prob[j] += torch.log(probs[j, input_ids_with_action[j, i]])
                        action_logits.append(total_log_prob) 
                        # sample from tempered policy
                    action_logits = torch.stack(action_logits) / pf_temp

                    action_logits = action_logits.to(torch.float32)

                    probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))
                

                    dist = Categorical(probs=probabilities.t())

                    idx = dist.sample()

                    action = allowed_actions_[idx].lower()
                
            else:
                action = random.choice(allowed_actions)
            if 'I have that, ' not in last_state:
                last_state_ = 'I have that, ' + last_state
                states.append(last_state_.split('I have that, ')[1].strip())
            else:
                states.append(last_state.split('I have that, ')[1].strip())

            actions.append(action)
            
            last_action = action

            if "Pick" in last_action or "Pick".lower() in last_action: 
                world_update_prompt = self.prompts["world_update_pickup"].format(last_state, last_action.capitalize())
            elif "Unstack" in last_action or "Unstack".lower() in last_action:
                world_update_prompt = self.prompts["world_update_unstack"].format(last_state, last_action.capitalize())
            elif "Put" in last_action or "Put".lower() in last_action:
                world_update_prompt = self.prompts["world_update_putdown"].format(last_state, last_action.capitalize())
            elif "Stack" in last_action or "Stack".lower() in last_action: 
                world_update_prompt = self.prompts["world_update_stack"].format(last_state, last_action.capitalize())
            

            if (last_state, last_action) in self.transitions:
                # if s, a, s' have been observed
                new_state = self.transitions[(last_state, last_action)]
            else:
                # if s, a, s' have not been observed, use World Model to predict the state and store it.
                lora_to_base(self.model)
                
                world_output = self.query_LM(self.model, self.world_tokenizer, world_update_prompt, do_sample=False, num_return_sequences=1,
                                    eos_token_id=eos_token_id)[0]
                world_change = world_output.split("[CHANGE]")[-1]
                new_state = apply_change(world_change, last_state)
                self.transitions[(last_state, last_action)] = new_state
            last_state = new_state
        if 'I have that, ' not in last_state:
            last_state_ = 'I have that, ' + last_state
            states.append(last_state_.split('I have that, ')[1].strip())
        else:
            states.append(last_state.split('I have that, ')[1].strip())
        goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal)
        meetings = [g in new_state for g in goals]
        if sum(meetings) == len(meetings):
            r1 = 100
        else:
            r1 = 10 * sum(meetings) / len(meetings)

        r1 = torch.tensor(r1).to(self.device)

        return None, actions, states, r1, None

    def local_search(self,
                            initial_state,
                            goal,
                            max_steps,
                            plan,
                            states, 
                            eos_token_id,
                            pf_temp=1.0,
                            mode="train",
                          ):
        """
        return: trajs, probability of each action in the trajs, log rewards of the trajs, log rewards of (state, action)
        """
        K = self.args.step // 2
        states = []
        actions = []
        if self.args.use_lora:
            base_to_lora(self.model)
        self.model.eval()
        prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=1)
        last_state = initial_state

        for step in range(max_steps):
            
            # epsilon greedy
            if step < K:
                action = plan[step]
            else:
                allowed_actions = generate_all_actions(last_state)
                allowed_actions_ = [act for act in allowed_actions if act.lower() not in actions]
                if len(allowed_actions_) != 0:
                    action = random.choice(allowed_actions_)
                else:
                    action = random.choice(allowed_actions)
                action = action.lower()
                

            if 'I have that, ' not in last_state:
                last_state_ = 'I have that, ' + last_state
                states.append(last_state_.split('I have that, ')[1].strip())
            else:
                states.append(last_state.split('I have that, ')[1].strip())

            actions.append(action)
            
            last_action = action

            if "Pick" in last_action or "Pick".lower() in last_action: 
                world_update_prompt = self.prompts["world_update_pickup"].format(last_state, last_action.capitalize())
            elif "Unstack" in last_action or "Unstack".lower() in last_action:
                world_update_prompt = self.prompts["world_update_unstack"].format(last_state, last_action.capitalize())
            elif "Put" in last_action or "Put".lower() in last_action:
                world_update_prompt = self.prompts["world_update_putdown"].format(last_state, last_action.capitalize())
            elif "Stack" in last_action or "Stack".lower() in last_action: 
                world_update_prompt = self.prompts["world_update_stack"].format(last_state, last_action.capitalize())
            

            if (last_state, last_action) in self.transitions:
                # if s, a, s' have been observed
                new_state = self.transitions[(last_state, last_action)]
            else:
                # if s, a, s' have not been observed, use World Model to predict the state and store it.
                lora_to_base(self.model)
                world_output = self.query_LM(self.model, self.world_tokenizer, world_update_prompt, do_sample=False, num_return_sequences=1,
                                    eos_token_id=eos_token_id)[0]
                world_change = world_output.split("[CHANGE]")[-1]
                
                new_state = apply_change(world_change, last_state)
                self.transitions[(last_state, last_action)] = new_state
            last_state = new_state
        if 'I have that, ' not in last_state:
            last_state_ = 'I have that, ' + last_state
            states.append(last_state_.split('I have that, ')[1].strip())
        else:
            states.append(last_state.split('I have that, ')[1].strip())
        goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal)
        meetings = [g in new_state for g in goals]
        if sum(meetings) == len(meetings):
            r1 = 100
        else:
            r1 = 10 * sum(meetings) / len(meetings)

        r1 = torch.tensor(r1).to(self.device)

        return None, actions, states, r1, None

    def forward_prob(self, goal, actions, states):
        if self.args.use_lora:
            base_to_lora(self.model)
        prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=1)

        initial_state = states[0]

        last_state = initial_state
        log_pf = []
        log_bf = []
        for step in range(len(actions)):
            
            icl_template = prompt["icl_list"][step // 2]
            icl_template = add_time(icl_template)
            previous_action = ""
            current_state = last_state
            allowed_actions = generate_all_actions(last_state)

            inputs = icl_template.replace("<init_state>", current_state.lstrip())\
                .replace("<goals>", goal).replace("<action>", previous_action.lstrip()).replace("<step>", str(step).strip()).strip()

            input_ids = self.tokenizer.encode(inputs.lstrip() + "\n", return_tensors='pt').to(self.device)
            
            action = actions[step]

            bsz = len(allowed_actions)  
            action_texts = [ac.lower() for ac in allowed_actions]
            action_ids = [self.tokenizer.encode(a, add_special_tokens=False, return_tensors='pt').to(self.device) for a in action_texts]
            
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

            probabilities = torch.exp(action_logits) / torch.sum(torch.exp(action_logits))

            idx = allowed_actions.index(action.capitalize())

            log_pf.append(torch.log(probabilities[idx]))
            
            if step < len(actions)-1:
                last_state = states[step+1]
            
            allowed_actions = generate_all_actions(last_state)
            pb = torch.tensor(1 / len(allowed_actions))
            log_bf.append(torch.log(pb))
        return torch.stack(log_pf).sum(), torch.stack(log_bf).sum()

    def get_ll_reward(self, actions, states, goal):

        reward = []

        prompt = sample_prompt(self.init_prompt, shuffle_prompt=False, num_shot=4)
        for step_idx, (state, action) in enumerate(zip(states, actions)):
            icl_template = prompt["icl_list"][step_idx // 2]
            if step_idx == 0:
                previous_action = ""
                current_state = state
            else:
                previous_action = actions[step_idx-1] + "\n"
                current_state = states[step_idx-1]
            inputs = icl_template.replace("<init_state>", current_state.lstrip())\
                .replace("<goals>", goal).replace("<action>", previous_action.lstrip())

            intuition = self.get_likelihood(inputs, [inputs + action.lstrip()])[0]
            self.ll_reward_dict[(step_idx, state, action, goal)] = intuition
            reward.append(intuition)

        return torch.tensor(reward).to(self.device)

    def get_likelihood(
            self,
            prefix: str,
            contents: list[str],
    ):
        lora_to_base(self.model)
        bsz = len(contents)
        prefix_tokens = self.world_tokenizer.encode(prefix, add_special_tokens=True)
        prompts_tokens = [self.world_tokenizer.encode(x, add_special_tokens=True) for x in contents]

        for prompt_tokens in prompts_tokens:
            assert prompt_tokens[: len(prefix_tokens)] == prefix_tokens

        max_prompt_size = max([len(t) for t in prompts_tokens])
        total_len = max_prompt_size
        tokens = torch.full((bsz, total_len), self.world_tokenizer.pad_token_id).cuda().long()

        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:2048].long()

        with torch.no_grad():
            outputs = self.model(tokens)
            logits = outputs.logits
        acc_probs = torch.zeros(bsz).cuda()
        for i in range(len(prefix_tokens), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.world_tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])

        return acc_probs

    def query_LM(self, worldmodel, tokenizer, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.7):
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

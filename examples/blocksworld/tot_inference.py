import pickle
from typing import Type, Callable, Optional

import numpy as np
from tqdm import tqdm
from datetime import datetime
import copy

from typing import NamedTuple
import reasoners.benchmark.bw_utils as utils

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners import WorldModel, LanguageModel, SearchConfig
from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import MCTS

BWAction = str
class BWState(NamedTuple):
    """The state of the Blocksworld for ToT

    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    action_history: list[str]
    end: bool


class BWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_candidate: int = 5) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate

    def get_actions(self, state: BWState) -> list[BWAction]:
        prompts = self.prompt["icl"] + "\n".join([""] + state.action_history + [""])
        ouputs = self.base_model.generate(prompts,
                                          num_return_sequences=self.n_candidate,
                                          max_length=20,
                                          eos_token_id="\n",
                                          hide_input=True)
        
        return [output.split("\n")[0] for output in ouputs]

    def reward(self, state: BWState, action: BWAction) -> tuple[float, dict]:
        
        icl_template = self.prompt["icl_list"][state.step_idx // 2]
        # every two step, we will deduct the icl prompt
        # so that the distribution of step length is more reasonable
        
        inputs = icl_template.replace("<init_state>", current_blocks_state)\
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True)).replace("<action>", previous_action)
        intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]

        self_eval_prompt = self.prompt["self-eval"].replace("<init_state>", current_blocks_state)\
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True)).replace("<action>", action)
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
            [self_eval_prompt + "good"])[0]

        return self.calculate_reward(intuition, self_eval), {'intuition': intuition, "self_eval": self_eval}

    def calculate_reward(self, intuition, self_eval, goal_reached=None):
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: BWState, action: BWAction,
               intuition: float = None,
               self_eval: float = None,
               goal_reached: tuple[bool, float] = None) -> float:
        assert intuition is not None, "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
        assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        return (self.calculate_reward(intuition, self_eval, goal_reached), 
                {'intuition': intuition, 'goal_reached': goal_reached})

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)

class BlocksWorldModel(WorldModel):
    """Blocks World World Model
    State: (step_idx, action_history: [str])
    Action: e.g. "put the red block on the green block"
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 6,
                 batch_size=2) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> BWState:
        """Initialize the world model.

        :return: the initial state
        """
        return BWState(step_idx=0, action_history=[], end=False)

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of BlocksWorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        if action != "[PLAN END]":
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history + [action], end=False)
        else:
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history, end=True)
        return state, {}

    def is_terminal(self, state: BWState) -> bool:
        if state.end:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

def dfs_bw(base_model: LanguageModel,
           prompt: dict,
           search_algo: Type[SearchAlgorithm] = MCTS,
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 6,
           reward_alpha: float = 0.5,
           batch_size = 1,
           goal_reached_reward = 100,
           goal_reward_default = 0.,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           domain_file: str = "",
           config_file: str = "",
           lm_plan_file: str = 'lm_plan.tmp',
           **search_algo_params):

    search_algo_params |= {"depth_limit": depth_limit}
    world_model = BlocksWorldModel(base_model=base_model, prompt=prompt, batch_size=batch_size, max_steps=depth_limit)
    config = BWConfig(base_model=base_model, prompt=prompt, batch_size=batch_size,
                      reward_alpha=reward_alpha, goal_reached_reward=goal_reached_reward,
                      goal_reward_default=goal_reward_default)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random
    import torch
    import torch.backends.cudnn
    from reasoners.lm import LlamaModel, Llama2Model
    from reasoners.lm.llama_model import DummyLLaMAModel
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    def exllama_main(
            model_dir = '/data/haotian/RAP_tune/Llama-2-13B-GPTQ',
            lora_dir = None,
            prompt_path: str = 'examples/blocksworld/prompts/prompt.json',
            data_path: str = 'examples/blocksworld/data/step_4.json',
            disable_log: bool = False,
            config_file: str = "examples/blocksworld/data/bw_config.yaml",
            domain_file: str = "examples/blocksworld/data/generated_domain.pddl",
            lm_plan_file: str = 'lm_plan.tmp',
            depth_limit: int = 6,
            batch_size: int = 1,
            mem_map = None,
            **kwargs
            ):
        print(model_dir)
        from reasoners.lm import ExLlamaModel  # Maybe other transformer models also support
        with open(prompt_path) as f:
            prompt = json.load(f)
        device = torch.device("cuda:0")
        """
        llama_model = ExLlamaModel(model_dir, 
                                   lora_dir, 
                                   device=device, 
                                   max_batch_size=batch_size, 
                                   max_new_tokens=200, 
                                   max_seq_length=2048, 
                                   mem_map=mem_map)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs
        """
        llama_model = DummyLLaMAModel(model_dir,
                                      0, 0)
        dfs_bw(llama_model,
               prompt,
               disable_log=disable_log,
               data_path=data_path,
               config_file=config_file,
               domain_file=domain_file,
               depth_limit=depth_limit,
               lm_plan_file=lm_plan_file,
               batch_size=batch_size, **kwargs)
    
    fire.Fire(exllama_main) # user will need to switch the model in the code

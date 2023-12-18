from reasoners.lm import ExLlamaModel
import json
import fire
import itertools
import os
from typing import Sequence, Any
import json
from tqdm import tqdm
import pickle
from typing import Type, Callable, Optional

from dataset import ProntoQADataset, ProntoQAProblem, ProntoQAExample
from reasoners import LanguageModel, SearchAlgorithm, Reasoner
import torch
import prompts.finish
import prompts.valid
import prompts.next_step

from reasoners import WorldModel, SearchConfig
from reasoners.lm import llama_cpp_model
from reasoners.visualization import visualize
from search_config import ProntoQAConfig
from reasoners.algorithm import MCTS, BeamSearch, DFS
from reasoners.benchmark import ProntoQAEvaluatorFinal

ProntoQAState = list[str]
ProntoQAAction = str

class ProntoQAToTWorldModel(WorldModel[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self) -> None:
        super().__init__()
    
    def init_state(self) -> ProntoQAState:
        return []
    
    def step(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[ProntoQAState, dict]:
        return state + [action], {}
    
    def is_terminal(self, state: ProntoQAState) -> bool:
        if len(state) > 0 and "The answer is" in state[-1]:
            return True
        return False
    
class ProntoQAToTSearchConfig(SearchConfig[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self, base_model, n_actions=5, temperature=0.8) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.temperature = temperature
        self.base_model = base_model
        assert temperature > 0, "Temperature = 0 indicates greedy decoding. There is no point running multiple chains"
    def get_actions(self, state: ProntoQAState) -> list[ProntoQAAction]:
        # print(f"state: {state}\n")
        input_prompt = self.prompt
        input_prompt += "Q: " + self.example.test_example.question + " " + self.example.test_example.query + "\nA:"
        # print(f"input_prompt: '{input_prompt}'\n")
        input_prompt += "".join([" " + s for s in state])
        output = self.base_model.generate([input_prompt] * self.n_actions, eos_token_id=29889, hide_input=True, temperature=self.temperature, do_sample=True).text
        ret = [o.strip() for o in output]
        # deduplicate
        ret = dict.fromkeys(ret).keys()
        return ret

    def fast_reward(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[float, dict]:
        input_prompt = self.prompt
        input_prompt += "Q: " + self.example.test_example.question + " " + self.example.test_example.query + "\nA:"
        input_prompt += "".join([" " + s for s in state])
        candidate = input_prompt + " " + action
        intuition = self.base_model.get_loglikelihood(input_prompt, 
            [candidate])[0]
        return intuition, {"intuition": intuition}

    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        intuition = kwargs["intuition"]
        return intuition, {"intuition": intuition}

def main(model_dir: str,
           search_algo: str = "beam",
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           temperature: float = 0.8,
           mem_map: str = None,
           **search_algo_params):

    if search_algo == "beam":
        search_algo_params |= {"max_depth": depth_limit}
    elif search_algo == "dfs":
        search_algo_params |= {"depth": depth_limit}
    else:
        print("Unknown search algorithm", search_algo)
        raise NotImplementedError
    

    def bfs_pronto_extractor(algo_output):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # to make sure the plan is saved before evaluation in multi-process setting
        try:
            answer = "\n".join(algo_output.terminal_node.state[2::2])
            answer = answer.replace("So ", "")
            return answer

        except Exception as e:
            print("Error in output extraction,", e)
            return ""
    
    def dfs_bw_extractor(algo_output):
        pass

    base_model = ExLlamaModel(model_dir, 
                              lora_dir=None, 
                              device=torch.device("cuda:0"), 
                              max_batch_size=1, 
                              max_new_tokens=200, 
                              max_seq_length=2048, 
                              mem_map=mem_map,
                              log_output=True)

    world_model = ProntoQAToTWorldModel()
    search_config = ProntoQAToTSearchConfig(base_model=base_model, temperature=temperature)
    
    output_extractor = dfs_bw_extractor if search_algo == "dfs" else bfs_pronto_extractor
    if search_algo == "dfs":
        search_algo = DFS(**search_algo_params)
    elif search_algo == "beam":
        search_algo = BeamSearch(**search_algo_params)
    else:
        raise NotImplementedError
   
    with open('examples/rap_prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)
    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/rap_prontoqa/data/345hop_random_true.json'
        ),
        output_extractor=output_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
    )

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    fire.Fire(main)

# CUDA_VISIBLE_DEVICES=0 python examples/rap_prontoqa/inference_tot.py --depth_limit 10 --model_dir $LLAMA2_CKPTS --beam_size 10 --temperature 0.8 --reward_aggregator mean --search_algo beam > debug_bfs.log
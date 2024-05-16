from reasoners.lm import ExLlamaModel
import json
import fire
from typing import Sequence, Any
import json
from tqdm import tqdm
from typing import Type, Callable, Optional, Literal

from dataset import ProntoQADataset, ProntoQAExample
from reasoners import Reasoner
import torch
import prompts.finish
import prompts.next_step
import prompts.valid_tot
import numpy as np
import random
from reasoners import WorldModel, SearchConfig
from reasoners.algorithm import MCTS, BeamSearch, DFS
from reasoners.benchmark import ProntoQAEvaluatorFinal

ProntoQAState = list[str]
ProntoQAAction = str

def remove_so_prefix(s):
    if s.startswith('So '):
        return s[3:]
    return s

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
        #eos_token_id=29889
        eos_token_id=["."]
        output = self.base_model.generate([input_prompt] * self.n_actions, eos_token_id=eos_token_id, hide_input=True, temperature=self.temperature, do_sample=True).text
        ret = [o.strip() for o in output]
        print(f"Input prompt to model.generate: {input_prompt}")
        print(f"model generated actions: {ret}")
        # deduplicate
        ret = dict.fromkeys(ret).keys()
        return ret

    def fast_reward(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[float, dict]:
        processed_state = [remove_so_prefix(s) for s in state]
        processed_action = remove_so_prefix(action)
        input_prompt = self.prompt
        input_prompt += "Q: " + self.example.test_example.question + " " + self.example.test_example.query + "\nA:"
        input_prompt += "".join([" " + s for s in processed_state])
        candidate = input_prompt + " " + processed_action
        intuition = self.base_model.get_loglikelihood(input_prompt, 
            [candidate])[0]
        
        print(f" prompt: {self.prompt}")
        print(f"action: {processed_action}")
        print(f"input_prompt: {input_prompt}")
        print("hello")
        print(f"state: {processed_state}")

        input_prompt = ""
        input_prompt += prompts.valid_tot.EXAMPLES
        input_prompt += prompts.valid_tot.FACTS_FORMAT.format(self.example.test_example.question or "", self.example.test_example.query)
        input_prompt += prompts.valid_tot.NEXT_STEP_FORMAT.format(',\n'.join(f'"{statement}"' for statement in processed_state))
        input_prompt += prompts.valid_tot.VALID_PREFIX

        output_logits = self.base_model.get_next_token_logits(
            input_prompt,
            candidates=["Yes", "No"]
        )

        print(f"input_prompt: {input_prompt}")
        reward: float = output_logits[0][0].item()
        reward:float = torch.softmax(torch.tensor(output_logits[0]), dim=0)[0].item()
        print(f" reward: {reward}")

        self_eval = reward  
        print(f" intuition: {intuition}, self_eval: {self_eval}")
        return intuition*0.5 + self_eval*0.5, {"intuition": intuition, "self_eval":self_eval}

    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        # how correct is this last action
        intuition = kwargs["intuition"]
        self_eval = kwargs["self_eval"]
        return intuition*0.5 + self_eval*0.5, {"intuition": intuition, "self_eval":self_eval}

def main(
           model_dir: str,
           base_lm: Literal[ 'llama2',' exllama', 'llama3']  = 'exllama',
           llama_size = "7B",
           batch_size = 4,
           search_algo: str = "beam",
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           temperature: float = 0.8,
           mem_map: str = [16, 22],
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
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # to make sure the plan is saved before evaluation in multi-process setting
        try:
            answer = "\n".join(algo_output.terminal_state[2::2])
            answer = answer.replace("So ", "")
            return answer

        except Exception as e:
            print("Error in output extraction,", e)
            return ""

    if base_lm in ['llama2', 'llama3']:    
        import torch
        import torch.backends.cudnn
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    if base_lm == 'llama2':
        from reasoners.lm import Llama2Model
        base_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_lm == 'llama3':
        from reasoners.lm import Llama3Model
        base_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
    else:
        from reasoners.lm import ExLlamaModel  # Maybe other transformer models also support
        base_model = ExLlamaModel(model_dir, 
                                lora_dir=None, 
                                device=torch.device("cuda:0"), 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=mem_map)

    world_model = ProntoQAToTWorldModel()
    search_config = ProntoQAToTSearchConfig(base_model=base_model, temperature=temperature)
    
    output_extractor = dfs_bw_extractor if search_algo == "dfs" else bfs_pronto_extractor
    if search_algo == "dfs":
        search_algo = DFS(**search_algo_params)
    elif search_algo == "beam":
        search_algo = BeamSearch(**search_algo_params)
    else:
        raise NotImplementedError
   
    with open('examples/CoT/prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)
    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/CoT/prontoqa/data/345hop_random_true.json'
        ),
        output_extractor=output_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
    )

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    fire.Fire(main)

# CUDA_VISIBLE_DEVICES=0 python examples/tot/prontoqa/inference_tot.py --depth_limit 10 --model_dir $LLAMA2_CKPTS --beam_size 10 --temperature 0.8 --reward_aggregator mean --search_algo beam > debug_bfs.log

# python examples/rap_prontoqa/tot_inference.py --depth_limit 10 --model_dir /data/yi/Llama-2-70B-GPTQ/ --total_states 10 --temperature 0.8 --search_algo dfs --max_per_state 3 > debug_dfs.log
    
    # TODO: 1) remove total state, depth limit 2) 
# python examples/tot/prontoqa/tot_inference.py --depth_limit 10 --model_dir /data/yi/Llama-2-70B-GPTQ/ --total_states 10 --temperature 0.8 --search_algo dfs --max_per_state 3 > debug_dfs.log

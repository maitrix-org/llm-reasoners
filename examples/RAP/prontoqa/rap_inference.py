import itertools
import os
import json
import fire

from dataset import ProntoQADataset
from reasoners import Reasoner

from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAAction
from reasoners.algorithm import MCTS
from reasoners.benchmark import ProntoQAEvaluatorFinal
def rap_answer_extractor(mcts_result):
    if mcts_result.trace is None:
        return ""
    else:
        return "\n".join([mcts_result.trace[0][i].body for i in range(1, len(mcts_result.trace[0]) - 1)])
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(base_model:str= "llama2",
           model_dir: str=  os.environ.get("LLAMA2_CKPTS", None) ,
           llama_size: str= "7B",
           batch_size: int= 1,
           mem_map: str = "[16, 22]",
           temperature: float = 0.8,
           n_candidates: int = 4,
           **search_algo_params):
    import numpy as np
    from reasoners.lm import ExLlamaModel , Llama2Model, Llama3Model
    if base_model == "llama2":
        language_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_model == "llama3":
        language_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_model == "exllama":
        language_model = ExLlamaModel(model_dir,
                                    lora_dir=None, 
                                    max_batch_size=1, 
                                    max_new_tokens=200, 
                                    max_seq_length=2048, 
                                    mem_map=mem_map,
                                    log_output=True)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs
    else:
        raise ValueError


    with open('examples/CoT/prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    world_model = ProntoQAWorldModel(base_model=language_model)
    search_config = ProntoQAConfig(base_model=language_model, temperature=temperature, n_candidates=n_candidates)
    search_algo = MCTS(output_trace_in_each_iter=True, cum_reward=np.mean, **search_algo_params)
    reasoner =  Reasoner(
            world_model=world_model,
            search_config=search_config,
            search_algo=search_algo
        )

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="rap",
        disable_log=False,
        output_extractor=rap_answer_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2]),
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/CoT/prontoqa/data/345hop_random_true.json'
        )
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4)
    print(f"accuracy: {accuracy}")


if __name__ == '__main__':
    fire.Fire(main)
# CUDA_VISIBLE_DEVICES=0,1 python examples/rap/prontoqa/rap_inference.py --mem_map "[16, 22]" --depth_limit 6 | tee debug_rap.log
    
# CUDA_VISIBLE_DEVICES=0,1 python examples/rap/prontoqa/rap_inference.py --mem_map "[16, 22]" --depth_limit 6 --w_exp 2 | tee debug_rap_2.log

# CUDA_VISIBLE_DEVICES=0,1 python examples/rap/prontoqa/rap_inference.py --mem_map "[16, 22]" --depth_limit 6 --n_candidates 1 --temperature 0.0 | tee debug_rap_chain.log
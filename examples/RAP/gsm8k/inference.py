from typing import Type, Callable, Optional, Literal

import numpy as np

from reasoners.benchmark import GSM8KEvaluator

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation

from world_model import GSM8kWorldModel, GSM8kPromptDict
from search_config import GSM8kConfig, GSM8kUsefulPrompt
import utils


def node_visualizer(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


def rap_gsm8k(
    base_model: LanguageModel,
    prompt: GSM8kPromptDict,
    useful_prompt: GSM8kUsefulPrompt,
    search_algo: Type[SearchAlgorithm] = MCTS,
    resume: int = 0,
    n_action: int = 4,
    n_confidence: int = 8,
    depth_limit: int = 5,
    force_terminating_on_depth_limit: bool = True,
    batch_size: int = 2,
    temperature: float = 0.8,
    early_stop_base: int = 2,
    early_stop_threshold: float = 0.5,
    reward_alpha: float = 0.5,
    reward_confidence_default: float = 0.8,
    cum_reward: Callable[[list[float]], float] = np.mean,
    calc_q: Callable[[list[float]], float] = max,
    log_dir: Optional[str] = None,
    disable_log: bool = False,
    disable_tqdm: bool = False,
    output_trace_in_each_iter: bool = True,
    aggregate: bool = True,
    **search_algo_params,
):

    if aggregate:
        aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy="edge")
    else:
        aggregator = None

    search_algo_params |= {
        "cum_reward": cum_reward,
        "calc_q": calc_q,
        "disable_tqdm": disable_tqdm,
        "output_trace_in_each_iter": output_trace_in_each_iter,
        "node_visualizer": node_visualizer,
        "aggregator": aggregator,
    }
    world_model = GSM8kWorldModel(
        base_model=base_model,
        n_confidence=n_confidence,
        batch_size=batch_size,
        temperature=temperature,
        early_stop_base=early_stop_base,
        early_stop_threshold=early_stop_threshold,
    )
    config = GSM8kConfig(
        base_model=base_model,
        useful_prompt=useful_prompt,
        n_actions=n_action,
        batch_size=batch_size,
        temperature=temperature,
        reward_alpha=reward_alpha,
        reward_confidence_default=reward_confidence_default,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        depth_limit=depth_limit,
    )
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=search_algo
    )

    evaluator = GSM8KEvaluator(
        output_extractor=utils.retrieve_answer,
        answer_extractor=utils.retrieve_answer_from_dataset,
        init_prompt=prompt,
        sample_prompt_type="rap",
        disable_log=disable_log,
        disable_tqdm=disable_tqdm,
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)


if __name__ == "__main__":
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA2_CKPTS", None)
    llama_3_ckpts = os.environ.get("LLAMA3_CKPTS", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")
        warnings.filterwarnings("ignore")

    def main(
        base_lm: Literal[
            "llama", "llama.cpp", "llama-2", "hf", "exllama", "llama-3"
        ] = "llama-3",
        llama_ckpts: str = llama_ckpts,
        llama_2_ckpts: str = llama_2_ckpts,
        llama_3_ckpts: str = llama_3_ckpts,
        llama_size: str = "13B",
        llama_cpp_path: str = None,
        llama_cpp_n_batch: int = 512,
        hf_path: str = "meta-llama/Llama-2-13b-hf",
        hf_peft_path: Optional[str] = None,
        hf_quantized: Optional[Literal["awq", "int8", "fp4", "nf4"]] = None,
        hf_load_awq_path: Optional[str] = None,
        exllama_model_dir: str = "WizardMath-13B-V1.0-GPTQ",
        exllama_lora_dir: Optional[str] = None,
        exllama_mem_map: Optional[str] = None,
        batch_size: int = 1,
        useful_prompt: str = "examples/RAP/gsm8k/prompts/useful_examples.json",
        prompt: str = "examples/RAP/gsm8k/prompts/prompt_pool.json",
        disable_log: bool = False,
        disable_tqdm: bool = False,
        **kwargs,
    ):

        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        with open(prompt) as f:
            prompt = json.load(f)
        if base_lm in ["llama", "llama-2", "llama-3"]:
            import torch
            import torch.backends.cudnn

            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        if base_lm == "llama":
            from reasoners.lm import LlamaModel

            base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == "llama.cpp":
            from reasoners.lm import LlamaCppModel

            base_model = LlamaCppModel(llama_cpp_path, n_batch=llama_cpp_n_batch)
        elif base_lm == "llama-2":
            from reasoners.lm import Llama2Model

            base_model = Llama2Model(
                llama_2_ckpts, llama_size, max_batch_size=batch_size, max_seq_len=4096
            )
        elif base_lm == "llama-3":
            from reasoners.lm import Llama3Model

            base_model = Llama3Model(
                llama_3_ckpts, llama_size, max_batch_size=batch_size, max_seq_len=4096
            )
        elif base_lm == "hf":
            from reasoners.lm import HFModel

            base_model = HFModel(
                hf_path,
                hf_path,
                max_batch_size=batch_size,
                max_new_tokens=512,
                peft_pth=hf_peft_path,
                quantized=hf_quantized,
                load_awq_pth=hf_load_awq_path,
            )
        elif base_lm == "exllama":
            from reasoners.lm import ExLlamaModel

            base_model = ExLlamaModel(
                exllama_model_dir,
                exllama_lora_dir,
                mem_map=exllama_mem_map,
                max_batch_size=batch_size,
                max_new_tokens=200,
                max_seq_length=3072,
            )
        elif base_lm == "openai":
            from reasoners.lm import OpenAIModel

            base_model = OpenAIModel(model="gpt-4o-mini", max_tokens=2048)
        else:
            assert False, f"cannot resolve {base_lm=}"

        rap_gsm8k(
            base_model=base_model,
            useful_prompt=useful_prompt,
            prompt=prompt,
            batch_size=batch_size,
            disable_log=disable_log or local_rank != 0,
            disable_tqdm=disable_tqdm or local_rank != 0,
            **kwargs,
        )

    fire.Fire(main)

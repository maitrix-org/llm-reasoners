import json
import fire
from typing import Sequence, Any
import json
import transformers

from dataset import ProntoQADataset
from reasoners.lm import ExLlamaModel, HFModel, BardCompletionModel, OpenAIModel, ClaudeModel, Llama2Model, Llama3Model
from reasoners.algorithm import MCTS
from reasoners.benchmark import ProntoQAEvaluatorFinal


class CoTReasoner():

    def __init__(self, base_model, n_sc=1, temperature=0.8, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs


    def __call__(self, example, prompt=None):
        # *base_facts, init_state = example.test_example.question.split(". ")
        # input_prompt += prompts.next_step.EXAMPLES
        input_prompt = prompt
        input_prompt += "Q: " + example.test_example.question + " " + example.test_example.query + "\nA:"
        print(f"input_prompt: '{input_prompt}'\n")

        if isinstance(self.base_model, OpenAIModel) or \
                isinstance(self.base_model, BardCompletionModel) or \
                isinstance(self.base_model, ClaudeModel):
            eos_token_id = []
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [108,109]
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama2Model):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama3Model):
            eos_token_id = ["\n", ".\n", ".\n\n"]
        elif self.base_model.model.config.architectures[0] == 'InternLM2ForCausalLM':
            eos_token_id = [364,402,512,756]
        elif self.base_model.model.config.architectures[0] == 'Qwen2ForCausalLM':
            eos_token_id = [198,271,382,624,151645]
        else:
            assert isinstance(self.base_model.model, transformers.LlamaForCausalLM)
            eos_token_id = [13]
        output = self.base_model.generate([input_prompt], eos_token_id=eos_token_id, hide_input=True, temperature=self.temperature, do_sample=True).text[0]
        print(f"output: '{output}'\n")
        
        steps = [s.split("So")[1].strip()+'.' for s in output.split('.') if "So" in s]
        
        return "\n".join(steps)

def main(base_model='exllama', model_dir=None, temperature=0.0, log_dir="name", quantized="int8", llama_size=None, batch_size=1):

    import torch, os
    import numpy as np
    from reasoners.lm import ExLlamaModel
    if base_model == 'exllama' and model_dir is None:
        print("Using Llama-2 70B by default")
        language_model = ExLlamaModel(os.environ['LLAMA2_CKPTS'],
                                    None, 
                                    max_batch_size=1, 
                                    max_new_tokens=200, 
                                    max_seq_length=2048, 
                                    mem_map=[16,22],
                                    log_output=True) #please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs
    else:
        if base_model == "google":
            language_model = BardCompletionModel("gemini-pro", additional_prompt="CONTINUE")
        elif base_model == "openai":
            language_model = OpenAIModel("gpt-4-1106-preview", additional_prompt="CONTINUE")
        elif base_model == "anthropic":
            language_model = ClaudeModel("claude-3-opus-20240229", additional_prompt="CONTINUE")
        elif base_model == 'llama2':
            language_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)   
        elif base_model == 'llama3':
            language_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
        elif base_model == "hf":
            language_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized)
        else:
            raise ValueError(f"Unknown model: {base_model}")
        # dataset = ProntoQADataset.from_file(
        #     'examples/prontoqa/data/345hop_random_true.json'
        # )

    with open('examples/CoT/prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner =  CoTReasoner(base_model=language_model, temperature=temperature)

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/CoT/prontoqa/data/345hop_random_true.json'
        ),
        output_extractor=lambda x: x,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4, log_dir=log_dir)
    print(f"accuracy: {accuracy}")

if __name__ == '__main__':
    fire.Fire(main)

# CUDA_VISIBLE_DEVICES=0,1 python examples/prontoqa/cot_inference.py --temperature 0.0 --log_dir "logs/prontoqa_llama2-70b-cot"
# CUDA_VISIBLE_DEVICES=3 python examples/prontoqa/cot_inference.py --model_dir "/data/haotian/RAP_tune/gemma-7b" --temperature 0.0 --log_dir "logs/prontoqa_gemma-cot"
# CUDA_VISIBLE_DEVICES=6 python examples/prontoqa/cot_inference.py --model_dir "/data/haotian/RAP_tune/Llama-2-13b-hf" --temperature 0.0 --log_dir "logs/prontoqa_llama2-13b-cot"
# CUDA_VISIBLE_DEVICES=7 python examples/prontoqa/cot_inference.py --model_dir "/data/haotian/RAP_tune/internlm2-7b" --temperature 0.0 --log_dir "logs/prontoqa_internlm2-7b-cot"
# CUDA_VISIBLE_DEVICES=4 python examples/prontoqa/cot_inference.py --model_dir "/data/haotian/RAP_tune/Mistral-7B-v0.1" --temperature 0.0 --log_dir "logs/prontoqa_mistral-7b-cot"
# CUDA_VISIBLE_DEVICES=4 python examples/prontoqa/cot_inference.py --model_dir "/data/haotian/RAP_tune/Qwen1.5-7B" --temperature 0.0 --log_dir "logs/prontoqa_qwen-7b-cot"
# CUDA_VISIBLE_DEVICES=3,4 python examples/prontoqa/cot_inference.py --model_dir "/data/haotian/RAP_tune/Mixtral-8x7B-v0.1" --temperature 0.0 --log_dir "logs/prontoqa_mixtral-cot" --quantized 'nf4'
# python examples/prontoqa/cot_inference.py --model_dir "google" --temperature 0.0 --log_dir "logs/prontoqa_gemini-cot"
# python examples/prontoqa/cot_inference.py --model_dir "openai" --temperature 0.0 --log_dir "logs/prontoqa_gpt-cot"
# python examples/prontoqa/cot_inference.py --model_dir "anthropic" --temperature 0.0 --log_dir "logs/prontoqa_claude-cot"
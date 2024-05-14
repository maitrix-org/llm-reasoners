from reasoners.lm import ExLlamaModel
import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
import utils
from typing import Literal
import fire
import transformers
class CoT_Reward_Reasoner():
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1, main_prompt = None, reward_model = None):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
        self.main_prompt = main_prompt
        self.reward_model = reward_model
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        question = self.main_prompt["prefix"].replace("{QUESTION}", example)
        outputs = []
        rewards = []
        do_sample = True
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Greedy decoding is not supported by HFModel. Using temperature = 1.0 instead.")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, BardCompletionModel) or isinstance(self.base_model, ClaudeModel):
            eos_token_id = []
        elif isinstance(self.base_model, ExLlamaModel):
            eos_token_id = [13]
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [108]
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        elif self.base_model.model.config.architectures[0] == 'InternLM2ForCausalLM':
            eos_token_id = [364,402,512,756]
        elif self.base_model.model.config.architectures[0] == 'Qwen2ForCausalLM':
            eos_token_id = [198,271,382,624,151645]
        else:
            assert isinstance(self.base_model.model, transformers.LlamaForCausalLM)###need to be modified for other model
            eos_token_id = [13]
        for i in range((self.n_sc - 1) // self.bs + 1):
            local_bs = min(self.bs, self.n_sc - i * self.bs)
            outputs += self.base_model.generate([inputs] * local_bs,
                                            hide_input=True,
                                            do_sample=do_sample,
                                            top_p=0.95,
                                            temperature=self.temperature,
                                            eos_token_id=eos_token_id).text
            #rewards for each solution
            rewards += self.reward_model.get_reward([question + out for out in outputs]).cpu().flatten().tolist()
        
        return [max(zip(rewards, outputs))[1]]

def main(base_lm:Literal['hf', 'google', 'openai', 'anthropic','exllama'],model_dir, lora_dir=None, mem_map=None, batch_size=1, prompt="examples/cot_gsm8k/prompts/cot.json", resume=0, log_dir=None, temperature=0, n_sc=1, quantized='int8', quantized_reward = None, reward_lm = None, reward_dir = None, device_map = None):

    if base_lm == "openai":
        base_model = OpenAIModel("gpt-4-1106-preview", additional_prompt="ANSWER")
    elif base_lm == "google":
        base_model = BardCompletionModel("gemini-pro", additional_prompt="ANSWER")
    elif base_lm == "anthropic":
        base_model = ClaudeModel("claude-3-opus-20240229", additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    elif base_lm == "exllama":
        base_model = ExLlamaModel(model_dir, max_batch_size = n_sc, max_seq_length=5000, max_new_tokens=1000, lora_dir=None)

    reward_tokenizer = None
    assert reward_dir is not None, "Reward model path cannot be none for CoT + Reward"
    if reward_lm == "openai" or reward_lm == "google" or reward_lm == "anthropic":
        raise NotImplementedError("{} model currently does not support reward model output",reward_lm)
    elif reward_lm == "hf":
        reward_model = HFModel(model_pth = reward_dir, tokenizer_pth = reward_dir,quantized=quantized_reward, is_reward_model=True, device_map = device_map)
    elif reward_lm == "exllama":
        reward_model = ExLlamaModel(reward_dir, max_batch_size = n_sc, max_seq_length=5000, lora_dir=None)
    
    
    
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoT_Reward_Reasoner(base_model, temperature=temperature, n_sc=n_sc, bs=batch_size, main_prompt = prompt, reward_model = reward_model)
    evaluator = GSM8KEvaluator(
                 output_extractor=utils.cot_sc_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot")
    from datetime import datetime
    log_dir =  f'logs/gsm8k_'\
                        f'cot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    if base_lm == 'hf':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f'_{model_name}'
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)
"""
CUDA_VISIBLE_DEVICES=0 python examples/reward_LM_gsm8k/inference.py \
    --model_dir /home/rahulc/Desktop/llama/llama-2-7b/quant/ \
    --reward_dir openbmb/Eurus-RM-7b \
    --base_lm exllama \
    --quantized_reward nf4 \
    --reward_lm hf \
    --batch_size 4 \
    --temperature 0.8 \
    --n_sc 4 \
    --device_map cuda:0
"""



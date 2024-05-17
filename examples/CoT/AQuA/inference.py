from typing import Literal
from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import AQuAEvaluator
import utils
import fire
from tqdm import tqdm
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
from reasoners.lm import  Llama2Model, Llama3Model
import transformers

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8, sc_num = 1):
        self.base_model = base_model
        self.temperature = temperature
        self.sc_num = sc_num
        
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        outputs = []
        do_sample = True
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, BardCompletionModel) or isinstance(self.base_model, ClaudeModel):
            eos_token_id = []
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [109] #Gemma use 109 for \n\n
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama2Model):
            eos_token_id = [13]
        elif isinstance(self.base_model, Llama3Model):
            eos_token_id = ["\n\n", ".\n", ".\n\n"]
        elif self.base_model.model.config.architectures[0] == 'InternLM2ForCausalLM':
            eos_token_id = [364, 402, 512, 756]#\n, \n\n, .\n, .\n\n
        elif self.base_model.model.config.architectures[0] == 'Qwen2ForCausalLM':
            eos_token_id = [198, 271, 382, 624, 151645]#same as above
        else:
            print(self.base_model.model.__class__)
            print(self.base_model.model.config.architectures[0])
            assert isinstance(self.base_model.model, transformers.LlamaForCausalLM)
            eos_token_id = [13]
        
        for _ in tqdm(range(self.sc_num), leave=False):
            
            output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=do_sample,
                                          temperature=self.temperature,
                                          eos_token_id=eos_token_id).text[0].strip()
            outputs.append(output)
        
        return outputs

def main(base_lm:Literal['hf', 'google', 'openai', 'anthropic','exllama','llama2'],
         model_dir= None, 
         llama_size=None,
         lora_dir = None, 
         mem_map = None, 
         batch_size=1, 
         prompt="examples/CoT/AQuA/prompts/cot.json", 
         data_path="examples/CoT/AQuA/data/", 
         datasetname="test",
         quantized='int8',
         resume=0, 
         temperature=0,
         sc_num=1,
         log_dir=None):

    if base_lm == "exllama":
        base_model = ExLlamaModel(model_dir, lora_dir,
                            mem_map=mem_map, max_batch_size=batch_size,
                            max_new_tokens=500, max_seq_length=2048)
    elif base_lm == "google":
        base_model = BardCompletionModel("gemini-pro", additional_prompt="ANSWER")
    elif base_lm == "openai":
        base_model = OpenAIModel("gpt-4-1106-preview", additional_prompt="ANSWER")
    elif base_lm == "anthropic":
        base_model = ClaudeModel("claude-3-opus-20240229", additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'llama2':
        base_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_lm == 'llama3':
        base_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, sc_num=sc_num)
    evaluator = AQuAEvaluator(
                 output_extractor=utils.retrieve_answer,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot",
                 dataset_path=data_path,
                 datasetname=datasetname)
    from datetime import datetime
    log_dir =  f'logs/AQuA_'\
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
CUDA_VISIBLE_DEVICES=0 python examples/cot/AQuA/inference_new.py \
--exllama_model_dir $MODEL_CKPTS --quantized 'int8'\
"""

"""
python examples/cot/AQuA/inference_new.py \
--base_lm google \
"""

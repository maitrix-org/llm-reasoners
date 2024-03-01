from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import AQuAEvaluator
import utils
import fire
from tqdm import tqdm
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import GPTCompletionModel
from reasoners.lm.gemini_model import BardCompletionModel
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
            print("Greedy decoding is not supported by HFModel. Using temperature = 1.0 instead.")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, GPTCompletionModel) or isinstance(self.base_model, BardCompletionModel):
            eos_token_id = []
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [109]###since here is \n\n in prompt
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        elif self.base_model.model.config.architectures[0] == 'InternLM2ForCausalLM':
            eos_token_id = [364,402,512,756]
        elif self.base_model.model.config.architectures[0] == 'Qwen2ForCausalLM':
            eos_token_id = [198,271,382,624,151645]
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

def main(exllama_model_dir= None, 
         exllama_lora_dir = None, 
         exllama_mem_map = None, 
         batch_size=1, 
         prompt="examples/AQuA_cot/prompts/cot.json", 
         quantized='int8',
         resume=0, 
         temperature=0,
         sc_num=1,
         log_dir=None,
         device_map=None):

    # base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir,
    #                       mem_map=exllama_mem_map, max_batch_size=batch_size,
    #                       max_new_tokens=500, max_seq_length=2048)
    if exllama_model_dir == "google":
        base_model = BardCompletionModel("gemini-pro")
    elif exllama_model_dir == "openai":
        base_model = GPTCompletionModel("gpt-4-1106-preview")
    else:
        base_model = HFModel(exllama_model_dir, exllama_model_dir,quantized=quantized)

    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, sc_num=sc_num)
    evaluator = AQuAEvaluator(
                 output_extractor=utils.retrieve_answer,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot")
    from datetime import datetime
    log_dir =  f'logs/AQuA'\
                        f'cot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    model_type = exllama_model_dir.split('/')[-1]
    log_dir = log_dir + f'_{model_type}'
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0


if __name__ == '__main__':
    fire.Fire(main)
    """
CUDA_VISIBLE_DEVICES=0 python examples/cot_gsm8k/inference_new.py \
--exllama_model_dir $MODEL_CKPTS --quantized 'int8'\
"""

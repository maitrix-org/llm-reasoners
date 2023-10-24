from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import AQuAEvaluator
import utils
import fire

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature,
                                          eos_token_id=[13]).text[0].strip()
        return output

def main(exllama_model_dir= '/data/haotian/RAP_tune/Llama-2-70B-GPTQ', 
         exllama_lora_dir = None, 
         exllama_mem_map = [16,22], 
         batch_size=1, 
         prompt="examples/AQuA_cot/prompts/cot.json", 
         resume=0, 
         temperature=0.8,
         log_dir=None):

    base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir,
                          mem_map=exllama_mem_map, max_batch_size=batch_size,
                          max_new_tokens=500, max_seq_length=2048)


    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature)
    evaluator = AQuAEvaluator(
                 output_extractor=utils.retrieve_answer,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot")

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)
    """
CUDA_VISIBLE_DEVICES=0,1 python examples/cot_gsm8k/inference_new.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]'
    """


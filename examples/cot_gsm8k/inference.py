from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import GSM8KEvaluator
import utils
import fire

class CoTReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0.8, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        outputs = []
        for i in range((self.n_sc - 1) // self.bs + 1):
            local_bs = min(self.bs, self.n_sc - i * self.bs)
            outputs += self.base_model.generate([inputs] * local_bs,
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature,
                                            eos_token_id=[13]).text
        return [o.strip() for o in outputs]

def main(exllama_model_dir, exllama_lora_dir, exllama_mem_map, batch_size=1, prompt="examples/cot_gsm8k/prompts/cot.json", resume=0, log_dir=None, temperature=0.8, n_sc=1):

    base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir,
                          mem_map=exllama_mem_map, max_batch_size=batch_size,
                          max_new_tokens=500, max_seq_length=2048)


    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, n_sc=n_sc, bs=batch_size)
    evaluator = GSM8KEvaluator(
                 output_extractor=utils.cot_sc_extractor,
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
CUDA_VISIBLE_DEVICES=2 python examples/cot_gsm8k/inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map None \
--temperature 0.0
    """

    """
CUDA_VISIBLE_DEVICES=1 python examples/cot_gsm8k/inference.py --exllama_model_dir $LLAMA2_CKPTS --exllama_lora_dir None --exllama_mem_map None --temperature 0.8 --n_sc 10 --log_dir logs/4-shot-cot-llama2-70b-sc-10-temp-0.8-speed --batch_size 4 | tee cot_sc.log
    """


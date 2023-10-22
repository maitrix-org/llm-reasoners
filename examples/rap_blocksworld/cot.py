from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import BWEvaluator
import utils
import fire

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature
    def __call__(self, example, prompt=None):
        inputs = prompt["icl"].replace("<init_state>", example["init"])\
            .replace("<goals>", example["goal"]).replace("<action>", "")
        output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature,
                                          eos_token_id='\n[').text[0][:-1].strip()
        return output

def main(exllama_model_dir, exllama_lora_dir, exllama_mem_map, data_path, prompt_path, disable_log=False, batch_size=1, config_file: str = "examples/rap_blocksworld/data/bw_config.yaml", domain_file: str = "examples/rap_blocksworld/data/generated_domain.pddl", resume=0, log_dir=None):

    base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir,
                          mem_map=exllama_mem_map, max_batch_size=batch_size,
                          max_new_tokens=300, max_seq_length=2048)


    with open(prompt_path) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=lambda x:x, sample_prompt_type="rap") # rap prompt includes cot
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)
    """
CUDA_VISIBLE_DEVICES=6,7 python examples/rap_blocksworld/cot.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v1/split_v1_step_4_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/rap_blocksworld_cot/"
    """


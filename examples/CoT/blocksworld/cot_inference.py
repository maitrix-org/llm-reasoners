from reasoners.lm import ExLlamaModel, HFModel
import json
from reasoners.benchmark import BWEvaluator
import fire
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
from reasoners.lm import  Llama2Model, Llama3Model
class CoTReasoner():
    def __init__(self, base_model, temperature=0.8, model_type="completion"):
        self.base_model = base_model
        self.temperature = temperature
        self.model_type = model_type

    def __call__(self, example, prompt=None):
        inputs = prompt["icl"].replace("<init_state>", example["init"])\
            .replace("<goals>", example["goal"]).replace("<action>", "")
        
        if self.model_type == "completion":
            output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature,
                                          eos_token_id='\n[').text[0][:-1].strip()
        elif self.model_type == "chat":
            output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=True,
                                          temperature=self.temperature).text[0].replace("[PLAN END]", "").strip()            
        return output

def main(model_dir, data_path, prompt_path, disable_log=False, batch_size=1, config_file: str = "examples/CoT/blocksworld/data/bw_config.yaml", domain_file: str = "examples/CoT/blocksworld/data/generated_domain.pddl", resume=0, log_dir=None, temperature=0.8, exllama_mem_map: str = None, quantized="int8", llama_path=None, llama_size=None):

    # base_model = ExLlamaModel(model_dir,
                        #   mem_map=exllama_mem_map, max_batch_size=batch_size,
                        #   max_new_tokens=300, max_seq_length=2048)

    
    if model_dir == "google":
        base_model = BardCompletionModel("gemini-pro", additional_prompt="CONTINUE")
    elif model_dir == "openai":
        base_model = OpenAIModel("gpt-4-1106-preview", additional_prompt="CONTINUE")
    elif model_dir == "anthropic":
        base_model = ClaudeModel("claude-3-opus-20240229", additional_prompt="CONTINUE")
    elif model_dir == 'llama2':
        base_model = Llama2Model(llama_path, llama_size, max_batch_size=batch_size)
    elif model_dir == 'llama3':
        base_model = Llama3Model(llama_path, llama_size, max_batch_size=batch_size)
    else:
        base_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized)
    with open(prompt_path) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(base_model, temperature=temperature, model_type="chat" if model_dir in ["openai", "google", "claude"] else "completion")
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=lambda x:x, sample_prompt_type="rap") # rap prompt includes cot
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)
from curses.ascii import GS
from reasoners.lm import ExLlamaModel
import json
from reasoners.lm.openai_model import GPTCompletionModel
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
import utils
import fire
import transformers
class CoTReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
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
                                            temperature=self.temperature,
                                            # eos_token_id=[13]).text #gemma的词表换了\n是108#\n\n是109
                                            eos_token_id=eos_token_id).text
        return [o.strip() for o in outputs]

def main(exllama_model_dir, exllama_lora_dir=None, exllama_mem_map=None, batch_size=1, prompt="examples/cot_gsm8k/prompts/cot.json", resume=0, log_dir=None, temperature=0, n_sc=1, quantized='int8'):

    if exllama_model_dir == "openai":
        base_model = GPTCompletionModel("gpt-4-1106-preview", additional_prompt="ANSWER")
    elif exllama_model_dir == "google":
        base_model = BardCompletionModel("gemini-pro", additional_prompt="ANSWER")
    else:
        base_model = HFModel(exllama_model_dir, exllama_model_dir, quantized=quantized)

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
--exllama_model_dir $Gemma_ckpts \ 这里gemma我用的非instruction tuning模型
"""



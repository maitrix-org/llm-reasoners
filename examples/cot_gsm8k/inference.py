from curses.ascii import GS
from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel
from sklearn import base
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
        if isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [108]
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM):
            eos_token_id = [13]
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

def main(exllama_model_dir, exllama_lora_dir=None, exllama_mem_map=None, batch_size=1, prompt="examples/cot_gsm8k/prompts/cot.json", resume=0, log_dir=None, temperature=0, n_sc=1):

    # base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir,
    #                       mem_map=exllama_mem_map, max_batch_size=batch_size,
    #                       max_new_tokens=500, max_seq_length=2048)
    base_model = HFModel(exllama_model_dir, exllama_model_dir,quantized='int8')

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
def calculate_acc():
    import pickle
    from datasets import load_dataset
    data = load_dataset('gsm8k','main','test')
    output_extractor=utils.cot_sc_extractor
    answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"])
    evaluator = GSM8KEvaluator(output_extractor=output_extractor,answer_extractor=answer_extractor,init_prompt=None,disable_log=False,disable_tqdm=False,sample_prompt_type="cot")
    correct_count = 0
    for i in range(1,1319):
        mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02232024-164842/algo_output/{i}.pkl', 'rb'))
        output = output_extractor(mcts_result)
        answer = answer_extractor(data['test'][i-1])
        correct = evaluator.eval_output(answer, output)
        correct_count += correct
        accuracy = correct_count / (i + 1)
    print(f'accuracy: {accuracy:.4f}')
if __name__ == '__main__':
    fire.Fire(main)
    # fire.Fire(calculate_acc)
    """
CUDA_VISIBLE_DEVICES=2 python examples/cot_gsm8k/inference.py \
--exllama_model_dir $Gemma_ckpts \ 这里gemma我用的非instruction tuning模型
"""



from reasoners.lm import ExLlamaModel
import json
from reasoners.lm.openai_model import GPTCompletionModel
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
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
        if isinstance(self.base_model, GPTCompletionModel) or isinstance(self.base_model, BardCompletionModel) or isinstance(self.base_model, ClaudeModel):
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
    elif exllama_model_dir == "anthropic":
        base_model = ClaudeModel("claude-3-opus-20240229", additional_prompt="ANSWER")
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

def calculate_acc():
    import pickle
    from datasets import load_dataset
    data = load_dataset('gsm8k','main','test')
    output_extractor=utils.retrieve_answer
    answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"])
    evaluator = GSM8KEvaluator(output_extractor=output_extractor,answer_extractor=answer_extractor,init_prompt=None,disable_log=False,disable_tqdm=False,sample_prompt_type="cot")
    correct_count = 0
    clean_path = '/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/03052024-061622/Claude3_new.json'
    import pandas as pd
    df = pd.read_json(clean_path, lines=True)
    cnt = 0
    df_c = pd.DataFrame(columns=['question', 'cot','index_ap'])
    for i in range(1,1319):
        mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/03052024-061622/algo_output/{i}.pkl', 'rb'))[-1]
        
        output_real = output_extractor(mcts_result)
        output_clean = df.loc[i-1,'metadata_generation'] + '.'
        print(output_clean)
        print(data['test'][i-1]['answer'])
        output = output_extractor(output_clean)
        answer = answer_extractor(data['test'][i-1])
        correct = evaluator.eval_output(answer, output)
        correct_real = evaluator.eval_output(answer, output_real)
        question = data['test'][i-1]['question']
        cot = mcts_result
        cot = cot.split('Q:')[0]
        cot_steps = cot.split('. ')
        cot_final = ""
        # cot_final = cot
        for j in range(len(cot_steps)):
            cot_final += f'Step {j+1}: ' + cot_steps[j] + ".\n"
        cot_final = cot_final.rstrip('\n')
        if correct_real != correct:
            # df_c.loc[cnt] = [question, cot_final, i-1]
            print(i)
            print(mcts_result)
            print(output_clean)
            print(answer)
            cnt += 1
        correct_count += correct
    accuracy = correct_count / (i + 1)
    print(cnt)
    print(f'accuracy: {accuracy:.4f}')
    # df_c.to_json('/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02292024-025642/cot_ap.json')
def fix_append():
    import pandas as pd
    ap_id_path = "/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02292024-025642/cot_ap.json"
    ap_path = "/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02292024-025642/GPT_4_ap.json"
    bug_path = "/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02292024-025642/GPT-4-turbo_cleaned.jsonl"
    df_id_ap = pd.read_json(ap_id_path)
    df_ap = pd.read_json(ap_path, lines=True)
    df_bug = pd.read_json(bug_path, lines=True)
    for i in range(len(df_ap)):
        bug_id = df_id_ap.loc[i,'index_ap']
        ap_metadata_generation = df_ap.loc[i,'metadata_generation']
        ap_text = df_ap.loc[i,'text']
        df_bug.loc[bug_id,'metadata_generation'] = ap_metadata_generation
        df_bug.loc[bug_id,'text'] = ap_text
    df_bug.to_json("/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02292024-025642/GPT-4-turbo_fixed.jsonl", orient='records', lines=True)

if __name__ == '__main__':
    # fire.Fire(main)
    fire.Fire(calculate_acc)
    # fire.Fire(fix_append)
    """
CUDA_VISIBLE_DEVICES=2 python examples/cot_gsm8k/inference.py \
--exllama_model_dir $Gemma_ckpts \ 这里gemma我用的非instruction tuning模型
"""



import json
from tqdm import tqdm
from reasoners.lm import Llama3Model
import utils
import fire

def prompt_input(question,prompt):
    return "Anser the question directly.\n" + prompt[0] + prompt[1] + prompt[2] + "Question: " + question + "\nAnser: "

def calculate_accuracy(predictions, references):
    correct = 0
    total = len(references)
    for pred, ref in zip(predictions, references):
        if pred.lower() == ref.lower():
            correct += 1
    return correct / total

def answer_extractor(output):
    start = output.find("1. ")
    if start != -1:
        end = output.find("2. ")
        if end == -1:
            return output[start+3:].strip()
        else:
            return output[start+3:end].strip()

    answer = output.strip() 
    return answer

def main(model_dir, llama_size="8B", prompt="examples/ReAct/hotpotqa/prompts/direct.json", data_path="examples/ReAct/hotpotqa/data/hotpot_dev_v1_simplified.json"):
    
    base_model = Llama3Model(model_dir, llama_size, max_batch_size=1, max_seq_len=20000)

    with open(data_path, 'r') as f:
        hotpotqa_data = json.load(f)[:1000]
    
    with open(prompt, 'r') as f:
        prompt = json.load(f)
    
    results = []
    answers = []
    for example in tqdm(hotpotqa_data):
        inputs = prompt_input(example['question'], prompt['react_pool'])
        outputs = base_model.generate([inputs],
                                    hide_input=True,
                                    do_sample=True,
                                    max_new_tokens=32,
                                    temperature=0, 
                                    eos_token_id=["\n",".\n","Question"]).text[0]
        # print(answer_extractor(outputs))
        results.append(answer_extractor(outputs))
        answers.append(example['answer'])

    print(calculate_accuracy(results,answers))
    

if __name__ == "__main__":
    fire.Fire(main)

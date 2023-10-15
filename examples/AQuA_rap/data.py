from datasets import Dataset
import os
import json
import re
from scipy import special
def data_reader(dataset,dataset_path, split=None, sample_size=100):
    questions = []
    answers = []
    options = []
    filename = os.path.join(dataset_path, 'AQuA.json')
    with open(filename, 'r') as file:
        lines = file.readlines()
        if split is not None:
            start, end = split
            lines = lines[start:end]
        for line in lines:
            data = json.loads(line)
            if isinstance(data, dict):
                options_list = data['options']
                options_dict = {}
                for option in options_list:
                    match = re.search(r'([A-E])\)[^0-9]*([\d.]+)', option)
                    if match:
                        options_dict[match.group(1)] = float(match.group(2))
                question_with_options = data['question'] + "\n" + "\n".join(data['options'])
                questions.append(question_with_options)
                # answers.append(options_dict.get(data['correct']))
                answers.append(data['correct'])
                options.append(options_list)
            else:
                raise ValueError("Unexpected data format")
    return Dataset.from_dict({"question": questions, "answer": answers, "options":options})

# dataset = data_reader("AQuA", '/data/haotian/RAP_tune/llm-reasoners/dataset/AQuA')
# print(dataset[0])
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("/data/haotian/RAP_tune/lora-gsm8k-1318-July-30-merged",legacy=False)
special_tokens = tokenizer.encode("\u221a",add_special_tokens=True)
print(tokenizer.decode([1723]))
print(special_tokens)
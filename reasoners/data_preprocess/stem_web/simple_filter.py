import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("TIGER-Lab/WebInstruct-verified", split="train")
def keep_example(example):
    return (
        example.get("category") != "Mathematics" and
        example.get("difficulty") in ["University", "PhD"] and
        example.get("answer_type") not in ["Boolean", "Multiple Choice"]
    )
filtered = dataset.filter(keep_example)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
answers = filtered["answer"]
encodings = tokenizer(answers, padding=False, truncation=False)
token_lengths = [len(ids) for ids in encodings["input_ids"]]

selected_indices = [i for i, length in enumerate(token_lengths) if length <= 30]
print(f"Number of examples with answer token length ≤ 30: {len(selected_indices)}")

samples = []
for idx in selected_indices:
    example = filtered[idx]
    ex_dict = dict(example)           
    ex_dict["token_length"] = token_lengths[idx]
    samples.append(ex_dict)

os.makedirs("output", exist_ok=True)
output_path = os.path.join("output", "samples_le30.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Saved {len(samples)} examples to {output_path}")
#!/usr/bin/env python3
# make_webinstruct_le30_parquet.py
"""
Convert samples_le30.json into a chat-style template and save as Parquet.
"""

import argparse, json, os
from typing import Dict, Any, List
import datasets
from transformers import AutoTokenizer
import pprint
import random
from verl.utils.data_process.utils import set_seed, save_dataset

# ------------------ prompt template definition ------------------ #
SYSTEM_MSG = (
    "You are a knowledgeable assistant. "
    "Answer the following questions and think step by step. Please output the final answer within \\boxed{}. "
)

def make_prompt(question: str) -> List[Dict[str, str]]:
    """
    Chat-style message list:
      - system: fixed instruction
      - user  : original question
    """
    return [
        {"role": "user",   "content": SYSTEM_MSG + question.strip()}
    ]

# --------------------------- main ------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="in_file",
                        default="samples_le30.json",
                        help="Input JSON file created earlier.")
    parser.add_argument("--out", dest="out_file",
                        default="webinstruct_le30.parquet",
                        help="Output Parquet filename.")
    parser.add_argument("--tokenizer",
                        default="Qwen/Qwen3-8B",
                        help="Name or path of tokenizer used for stats.")
    args = parser.parse_args()

    # 1. load json
    with open(args.in_file, encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    # 2. wrap into new template
    processed = []
    for ex in raw:
        question = ex["question"]
        answer   = ex["answer"]
        processed.append({
            "data_source": "WebInstruct-le30",
            "prompt": make_prompt(question),
            "raw_prompt": question,
            "ability": "QA",
            "apply_chat_template": True,
            "response": answer,
            "reward_model": {"ground_truth": answer},
            "extra_info": {
                "category":    ex.get("category"),
                "difficulty":  ex.get("difficulty"),
                "answer_type": ex.get("answer_type"),
                "token_length": ex["token_length"]
            },
        })
    # ---------- NEW: print a few samples -----------------
    print("\n*** Example prompts (3 random rows) ***")
    for sample in random.sample(processed, k=min(3, len(processed))):
        pprint.pprint(
            {
                "prompt": sample["prompt"],
                "response": sample["response"],
                "extra_info": sample["extra_info"]
            },
            compact=True,
            width=120
        )
        print("-" * 80)
    # ------------------------------------------------------

    # 3. build Dataset and save parquet
    ds = datasets.Dataset.from_list(processed)
    save_path = save_dataset(dataset=ds,
                             output_dir="./",
                             filename_prefix="stem",
                             sample_size=len(ds))
    
    print(f"Saved {len(ds)} rows to {save_path}")


    # 4. (optional) token-length stats over *prompt* messages
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    lengths = [
        len(tok.encode(msg["content"], add_special_tokens=False))
        for row in ds
        for msg in row["prompt"]
        if msg["role"] == "user"
    ]
    import numpy as np
    lengths = np.asarray(lengths)
    print(f"User-prompt token stats — min:{lengths.min()}, "
          f"max:{lengths.max()}, mean:{lengths.mean():.1f}, "
          f"std:{lengths.std():.1f}")

if __name__ == "__main__":
    main()

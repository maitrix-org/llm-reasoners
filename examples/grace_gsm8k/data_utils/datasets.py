import json
import os
import re
import torch as th
import sys 
sys.path.append('../')
from constants import A_DELIM, Q_DELIM


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True, decoder_only=True, add_delim=True):
        self.step_delimiter = " | "
        
        for ex in examples:
            if add_delim:
                ex["question"] = " ".join([Q_DELIM, ex["question"]])
                ex["answer"] = " ".join([A_DELIM, ex["answer"]])
            
            ex["question"] = " ".join(ex["question"].split())
            ex["answer"] = re.sub(r"\n+", "\n", ex["answer"])
            ex["answer"] = ex["answer"].replace(" \n", " \n").replace(".\n",
            "\n")
            ex["answer"] = ex["answer"].replace("\n", self.step_delimiter)
            assert self.step_delimiter in ex["answer"], "{} should be {}".format(self.step_delimiter, ex["answer"])

            ex.update(question=ex["question"])
            ex.update(answer=ex["answer"])
        
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]

        ## remove whitespaces
        self.qns = [" ".join(qn.split()) for qn in self.qns]
        self.ans = [" ".join(ans.split()) + tokenizer.eos_token for ans in self.ans]

        print("Example data question: ", self.qns[0])
        print("Example data answer: ", self.ans[0])
        self.qns = tokenizer(self.qns, padding=False, add_special_tokens=False)
        self.ans = tokenizer(self.ans, padding=False)

        self.loss_on_prefix = loss_on_prefix
        self.decoder_only = decoder_only
        self.tokenizer = tokenizer

        if decoder_only:
            self.max_len = max(
                [
                    len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                    for i in range(len(self.examples))
                ]
            )
            print(f"Max tokens: {self.max_len}")

        else: 
            self.max_input_len = max(
                [
                    len(self.qns["input_ids"][i])
                    for i in range(len(self.examples))
                ]
            )
            self.max_output_len = max(
                [
                    len(self.ans["input_ids"][i])
                    for i in range(len(self.examples))
                ]
            )
            print(f"Max input tokens: {self.max_input_len}")
            print(f"Max output tokens: {self.max_output_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        if self.decoder_only:
            pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
            tokens = qn_tokens + ans_tokens + pad_tokens
            attention_mask = (
                ([1] * len(qn_tokens)) 
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
            )
            tokens = th.tensor(tokens)
            attention_mask = th.tensor(attention_mask)
            labels = tokens.clone()
            if not self.loss_on_prefix:
                labels[: len(qn_tokens)] = -100
            return dict(input_ids=tokens, attention_mask=attention_mask, labels=labels)
        else: 
            tokens = qn_tokens # + input_pad_tokens
            labels = ans_tokens # + output_pad_tokens
            tokens = th.tensor(tokens)
            labels = th.tensor(labels)
            return dict(input_ids=tokens, labels=labels)

class ASDIVDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True, decoder_only=True, add_delim=True):
        
        self.step_delimiter = ";"
        for ex in examples:
            if add_delim:
                ex["question"] = " ".join([Q_DELIM, ex["question"]])
                ex["answer"] = " ".join([A_DELIM, ex["answer"]])
            
            ex["question"] = " ".join(ex["question"].split())
            ex["answer"] = re.sub(r"\n+", "\n", ex["answer"])
            ex["answer"] = ex["answer"].replace("\n", self.step_delimiter)
            ## replace "answer" in ex["answer"] with "ans" to make it consistent with the other datasets
            ex["answer"] = ex["answer"].replace("answer", "ans")
            assert self.step_delimiter in ex["answer"], "{} should be {}".format(self.step_delimiter, ex["answer"])
            ex.update(question=ex["question"])
            ex.update(answer=ex["answer"])

        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]

        self.ans = [a + tokenizer.eos_token for a in self.ans]
        print("Example data question: ", self.qns[0])
        print("Example data answer: ", self.ans[0])
        self.qns = tokenizer(self.qns, padding=False, add_special_tokens=False)
        self.ans = tokenizer(self.ans, padding=False)

        self.loss_on_prefix = loss_on_prefix
        self.decoder_only = decoder_only
        self.tokenizer = tokenizer

        if decoder_only:
            self.max_len = max(
                [
                    len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                    for i in range(len(self.examples))
                ]
            )
            print(f"Max tokens: {self.max_len}")

        else: 
            self.max_input_len = max(
                [
                    len(self.qns["input_ids"][i])
                    for i in range(len(self.examples))
                ]
            )
            self.max_output_len = max(
                [
                    len(self.ans["input_ids"][i])
                    for i in range(len(self.examples))
                ]
            )
            print(f"Max input tokens: {self.max_input_len}")
            print(f"Max output tokens: {self.max_output_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        if self.decoder_only:
            pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
            tokens = qn_tokens + ans_tokens + pad_tokens
            attention_mask = (
                ([1] * len(qn_tokens)) 
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
            )
            tokens = th.tensor(tokens)
            attention_mask = th.tensor(attention_mask)
            labels = tokens.clone()
            if not self.loss_on_prefix:
                labels[: len(qn_tokens)] = -100
            return dict(input_ids=tokens, attention_mask=attention_mask, labels=labels)
        else: 
            tokens = qn_tokens # + input_pad_tokens
            labels = ans_tokens # + output_pad_tokens
            tokens = th.tensor(tokens)
            labels = th.tensor(labels)
            return dict(input_ids=tokens, labels=labels)

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer
import openai
import backoff 

from .. import LanguageModel, GenerateOutput

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


class GPTModel(LanguageModel):
    def __init__(self, model='gpt-3.5-turbo', max_seq_len=2048, local_rank=-1, world_size=-1):
        super().__init__()
        self.model = model
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def generate(self, prompt, temperature=0.8, max_tokens=1000, generation_num=1, end_token=None) -> list:
        messages = [{"role": "user", "content": prompt}]
        return self.chatgpt(messages, temperature=temperature, max_tokens=max_tokens, generation_num=generation_num, end_token=end_token)

    @torch.no_grad()
    def chatgpt(self, messages, temperature=0.8, max_tokens=1000, generation_num=1, end_token=None) -> list:
        global completion_tokens, prompt_tokens
        outputs = []
        while generation_num > 0:
            cnt = min(generation_num, 20)
            generation_num -= cnt
            res = completions_with_backoff(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=end_token)
            outputs.extend([choice["message"]["content"] for choice in res["choices"]])
            # log completion tokens
            completion_tokens += res["usage"]["completion_tokens"]
            prompt_tokens += res["usage"]["prompt_tokens"]
        return outputs
    

    
    @torch.no_grad()
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        
        return []

    @torch.no_grad()
    def get_ll(
            self,
            prefix: str,
            contents: list[str],
    ) -> np.ndarray:

        return ['']
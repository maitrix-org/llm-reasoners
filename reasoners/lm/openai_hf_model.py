import os
import openai
import numpy as np
from typing import Optional, Union
import time
from .. import LanguageModel, GenerateOutput
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
import torch
from peft import PeftModel
import warnings
import copy
import sys
from optimum.bettertransformer import BetterTransformer

class GPT_HF_Model(LanguageModel):
    def __init__(self, openai_model:str, model_pth, tokenizer_pth, device, max_tokens:int = 2048, temperature=0.7, quantized=None, peft_pth=None, max_batch_size=1):
        self.openai_model = openai_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_batch_size = max_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, legacy=False)
        if quantized == "int8":
            self.model = LlamaForCausalLM.from_pretrained(
                model_pth,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",                                    
            )
        elif quantized == "nf4" or quantized  == "fp4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=quantized,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            print("quantizing.............................")

            self.model = LlamaForCausalLM.from_pretrained(
                model_pth,
                quantization_config=bnb_config,
                device_map="auto",
            )
        

        if peft_pth is not None:
            self.model = PeftModel.from_pretrained(
                self.model, 
                peft_pth,
                torch_dtype=torch.float16
            )
        
        self.device = device
        self.model = BetterTransformer.transform(self.model)
        self.model.eval()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2


        API_KEY = os.getenv("OPENAI_API_KEY", None)
        if API_KEY is None:
            raise ValueError("OPENAI_API_KEY not set, please run `export OPENAI_API_KEY=<your key>` to ser it")
        else:
            openai.api_key = API_KEY
    
    def generate(self,
                inputs: list[str],
                max_length: int = None,
                max_new_tokens: int = None,
                do_sample: bool = False,
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 20,
                eos_token_id: Union[None, str, list[str]] = None,
                logprobs: Optional[int] = None,
                temperature = None,
                hide_input: bool = True,
                output_log_probs: bool = False,
                **kwargs) -> GenerateOutput:
        
        gpt_temperature = self.temperature if temperature is None else temperature

        ###renaming
        max_tokens = max_length
        if max_tokens is None:
            max_tokens = self.max_tokens
        if max_new_tokens is not None:
            warnings.warn("max_new_tokens is deprecated, please use max_length instead")
            max_tokens = max_new_tokens
        if logprobs is None:
            logprobs = 0
        if num_return_sequences == 1 and len(inputs) > 0:
            if len(inputs) > 1:
                assert inputs[0] == inputs[1]#all should be the same
            num_return_sequences = len(inputs)
            inputs = inputs[0]
        if output_log_probs:
            warnings.warn("output_log_probs is not implemented yet")

        i = 1

        for i in range(1, 5):  # try 64 times ?
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if ('gpt-3.5-turbo' in self.openai_model) or ('gpt-4' in self.openai_model):
                    messages = [{"role": "user", "content": inputs}]
                    response = openai.ChatCompletion.create(
                        model=self.openai_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=gpt_temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=eos_token_id,
                    )
                    for choice in response["choices"]:
                        with open('openai_output.txt', 'a+') as f:
                            f.write(f"{choice['message']['content']}\n")
                    
                    return GenerateOutput(
                        text=[choice["message"]["content"] for choice in response["choices"]],
                        log_prob=None
                    )
                else:
                    response = openai.Completion.create(
                        model=self.openai_model,
                        prompt=inputs,
                        max_tokens=max_tokens,
                        temperature=gpt_temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=eos_token_id,
                        logprobs=logprobs,
                        **kwargs
                    )

                    return GenerateOutput(
                        text=[choice["text"] for choice in response["choices"]],
                        log_prob=[choice["logprobs"] for choice in response["choices"]]
                    )
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i*10} seconds")
                time.sleep(i*10)
        
        # after 64 tries, still no luck
        raise RuntimeError("GPTCompletionModel failed to generate output, even after 64 tries")
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)
        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[1] if len(token) > 1 else token[0])
        

        bsz = len(prompt)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)

        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            all_logits = self.model(**tokens, return_dict=True).logits[:,-1,:].squeeze(1)

        logits = []
        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())
        return logits

    @torch.no_grad()
    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        bsz = len(contents)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)
        prompts_tokens = self.tokenizer(contents, return_tensors='pt',add_special_tokens=False, padding=True).to(self.device)
        prefix_tokens = self.tokenizer(prefix, return_tensors='pt',add_special_tokens=False, padding=True).input_ids[0].to(self.device)
        
        for prompt_tokens in prompts_tokens.input_ids:
            assert torch.all(prompt_tokens[: len(prefix_tokens)] == prefix_tokens), (prompt_tokens, prefix_tokens)

        tokens = prompts_tokens
        logits = self.model(**tokens, return_dict=True).logits
        tokens = prompts_tokens.input_ids
        acc_probs = torch.zeros(bsz).to(self.device)
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i-1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        return acc_probs.cpu().numpy()
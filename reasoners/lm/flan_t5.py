from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
import os
import sys
from typing import Tuple, Union, Optional
import warnings
import random
import copy
import glob
import time

import torch
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download

from reasoners import LanguageModel, GenerateOutput

class FlanT5Model:
    def __init__(self, model_name='google/flan-t5-large'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(
            self,
            inputs: list[str],
            max_length: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 1.0,
            num_return_sequences: int = 1,
            eos_token_id: Union[None, str, int, list[Union[str, int]]] = None,
            hide_input: bool = True,
            output_log_probs: bool = False,
            **kwargs,
    ) -> GenerateOutput:
        if max_length is not None:
            print("max_length is not supported by FlanT5Model for generation. Use max_new_tokens instead.")
        
        # Handle eos_token_id
        if eos_token_id is not None:
            if isinstance(eos_token_id, (str, int)):
                eos_token_ids = [self.tokenizer.convert_tokens_to_ids(eos_token_id)] if isinstance(eos_token_id, str) else [eos_token_id]
            elif isinstance(eos_token_id, list):
                eos_token_ids = [self.tokenizer.convert_tokens_to_ids(token) if isinstance(token, str) else token for token in eos_token_id]
            else:
                raise ValueError("eos_token_id must be None, str, int, or list of str/int")
        else:
            eos_token_ids = [self.tokenizer.eos_token_id]

        # Prepare inputs
        encoded_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.model.config.max_length).to(self.device)
        input_ids = encoded_inputs['input_ids']

        # Generate outputs
        generated_sequences = []
        for _ in range(num_return_sequences):
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length or self.model.config.max_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_ids[0],  # Assumes single EOS token ID for simplicity
                **kwargs
            )
            generated_sequences += [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        
        if hide_input:
            generated_sequences = [seq[len(input_text):] for seq, input_text in zip(generated_sequences, inputs*num_return_sequences)]
        
        # Log probabilities are not directly supported in this implementation
        log_probs = None if not output_log_probs else [0] * len(generated_sequences)  # Placeholder for actual log probabilities

        return GenerateOutput(generated_sequences, log_probs)

    
    @torch.no_grad()
    def get_next_token_logits(self, prompt: Union[str, list[str]], candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        if isinstance(prompt, str):
            prompt = [prompt]
        # Ensure candidates is a list of lists
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)

        input_ids = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).input_ids
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

        # Select logits for the last token in each sequence
        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        candidate_logits = []

        for i, candidate_group in enumerate(candidates):
            candidate_group_logits = []
            for candidate in candidate_group:
                candidate_ids = self.tokenizer(candidate, add_special_tokens=False).input_ids
                # Assuming candidate is a single token, get its logit
                if len(candidate_ids) == 1:
                    logit = next_token_logits[i, candidate_ids[0]].cpu().numpy()
                else:
                    raise ValueError(f"Candidate '{candidate}' does not correspond to a single token.")
                candidate_group_logits.append(logit)
            candidate_logits.append(np.array(candidate_group_logits))

        return candidate_logits

    def get_loglikelihood(self, prompts, responses):
        # Approximation: compute the likelihood of the response given the prompt
        loglikelihoods = []
        for prompt, response in zip(prompts, responses):
            input_text = prompt + response
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_seq_len, truncation=True, padding="max_length")
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            # Compute log likelihood for response tokens
            response_ids = self.tokenizer(response, return_tensors="pt", input_ids=True).input_ids.to(self.device)
            shifted_logits = logits[:, :-1]
            log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
            response_log_probs = torch.gather(log_probs, 2, response_ids[:, :, None]).squeeze(-1)
            loglikelihood = response_log_probs.sum(-1)
            loglikelihoods.append(loglikelihood.item())
        return loglikelihoods

# Example Usage
if __name__ == "__main__":
    flan_t5_model = FlanT5Model()
    input_text = "translate English to French: The science of today is the technology of tomorrow."
    generated_texts = flan_t5_model.generate(input_text, num_return_sequences=1)
    for text in generated_texts:
        print(text)

    # For get_next_token_logits and get_loglikelihood, you would call them with appropriate inputs,
    # but note that their implementations here are simplified and might need adjustment for specific use cases.

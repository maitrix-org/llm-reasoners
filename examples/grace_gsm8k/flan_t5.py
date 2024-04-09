from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoTokenizer
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
from torch.nn.functional import log_softmax
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download

from reasoners import LanguageModel, GenerateOutput

class FlanT5Model:
    def __init__(self, model_name='mkhalifa/flan-t5-large-gsm8k'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def get_log_probs(self, inputs: list[str], generated_sequences: list[str]):
        log_probs = []
        for input_text, gen_seq in zip(inputs, generated_sequences):
            # print(f" inside logprobs")
            input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.model.config.max_length).input_ids.to(self.device)
            gen_ids = self.tokenizer(gen_seq, return_tensors="pt", truncation=True, padding="max_length", max_length=self.model.config.max_length).input_ids.to(self.device)
            
            outputs = self.model(input_ids=input_ids, labels=gen_ids)
            logits = outputs.logits  # [batch_size, sequence_length, config.vocab_size]
            
            # Shift logits and labels to align for calculating log_probs of generated tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = gen_ids[..., 1:].contiguous()
            
            # Flatten the logits and labels to calculate log_probs easily
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_labels = shift_labels.view(-1)
            
            # Calculate log probabilities
            log_probs_seq = log_softmax(flat_shift_logits, dim=1)
            flat_log_probs = log_probs_seq.gather(dim=1, index=flat_shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Reshape log_probs back to [batch_size, sequence_length]
            log_probs.append(flat_log_probs.view(shift_logits.size(0), shift_logits.size(1)).sum(1))
        
        return log_probs

    def generate(
            self,
            inputs: list[str],
            max_length: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            do_sample: bool = True,
            temperature: float = 0.8,
            top_k: int = None,
            top_p: float = 0.95,
            num_return_sequences: int = 1,
            eos_token_id: Union[None, str, int, list[Union[str, int]]] = None,
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
        model_inputs = inputs[0]  
        cur_prefix = inputs[1]
        
        model_encoded_inputs = self.tokenizer(model_inputs, return_tensors="pt").to(self.device)
        model_input_ids = model_encoded_inputs['input_ids']
        
        if len(cur_prefix)==0:
            cur_prefix_input_ids = torch.LongTensor([[self.tokenizer.pad_token_id]] * 1).to(self.device)
        else:
            cur_prefix_encoded_inputs = self.tokenizer(cur_prefix, return_tensors="pt").to(self.device)
            cur_prefix_input_ids = cur_prefix_encoded_inputs['input_ids']
            cur_prefix_input_ids = cur_prefix_input_ids[:, :-1]
            cur_prefix_input_ids= torch.cat((torch.LongTensor([[self.tokenizer.pad_token_id]] * 1).to(self.device), cur_prefix_input_ids), dim=1)


        input_generated_sequences = []
        input_generated_sequences += [self.tokenizer.decode(g, skip_special_tokens=True) for g in model_input_ids]
        input_generated_sequences = []
        input_generated_sequences += [self.tokenizer.decode(g, skip_special_tokens=True) for g in cur_prefix_input_ids]
       
 

        # Generate outputs
        generated_sequences = []
        total_generated_ids = []

        outputs = self.model.generate(
            decoder_input_ids=cur_prefix_input_ids ,
            input_ids=model_input_ids,
            attention_mask=model_input_ids.new_ones(model_input_ids.shape),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=1.0,
            return_dict_in_generate=True,
            num_return_sequences=20,
            # bad_words_ids = [[0]],
            no_repeat_ngram_size=0,
            sample_calc=True,
            temperature=temperature,
            output_scores=True,
            top_k=None,
            top_p=0.95,
            eos_token_id=[1820],  # Assumes single EOS token ID for simplicity
            tokenizer=self.tokenizer,
            **kwargs
        )
        output_sequences = outputs.sequences

        new_sequences = output_sequences[:, cur_prefix_input_ids.shape[1]-1:]
        transition_scores = self.model.compute_transition_scores(
                    outputs.sequences,
                    outputs.scores,
                    normalize_logits=True)# batch x seq

        ## normalize by length: exp()
        probs = torch.exp(transition_scores) # batch x seq
        logprobs = torch.log(probs) # batch x seq
        ## divide by length of each sequence
        seq_lens = torch.sum(new_sequences!= self.tokenizer.pad_token_id, dim=-1).unsqueeze(-1) # batch x 1
        logprobs = logprobs / seq_lens # batch x seq
        ### set -inf to 0
        logprobs[logprobs == float('-inf')] = 0.0
        seq_scores = torch.exp(torch.sum(logprobs, dim=-1))


        for generated_ids in new_sequences:
            generated_ids = generated_ids.view(1, -1)
            total_generated_ids.append(generated_ids)
            generated_sequences+=[self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]


        return GenerateOutput(generated_sequences, seq_scores)

    
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
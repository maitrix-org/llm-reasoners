import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple, Union, Optional
import copy

import numpy as np
import torch
import torch.distributed

import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_model_parallel_rank,
)

from llama3.model import ModelArgs, Transformer
from llama3.tokenizer import ChatFormat, Dialog, Message, Tokenizer
from reasoners import LanguageModel, GenerateOutput


class Llama3Model(LanguageModel):
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama3Model":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return model, tokenizer

    def __init__(self, path, size, max_batch_size=1, max_seq_len=2048, **kwargs):
        super().__init__()
        print(path, size, max_batch_size, max_seq_len)
        self.model, self.tokenizer = self.build(
            os.path.join(path, f"Meta-Llama-3-{size.upper()}"),
            os.path.join(path, f"Meta-Llama-3-{size.upper()}", "tokenizer.model"),
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **kwargs)
        self.max_seq_len = max_seq_len
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    @torch.inference_mode()
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 0.6,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 **kwargs) -> GenerateOutput:
        
        if max_new_tokens is None:
            max_new_tokens = self.model.params.max_seq_len - 1

        if not do_sample:
            if temperature != 0.6 and self.local_rank == 0:  # temperature is explicitly set with do_sample=False 
                warnings.warn('temperature is set, but do_sample=False, so temperature will be ignored.')
            temperature = 0

        eos_token_id_input = copy.deepcopy(eos_token_id)
        eos_token_id = []
        if eos_token_id_input is not None:
            if not isinstance(eos_token_id_input, list):
                eos_token_id_input = [eos_token_id_input]
            for token in eos_token_id_input:
                if isinstance(token, str):
                    tokenized = self.tokenizer.encode(token, bos=False, eos=False)
                    if len(tokenized) != 1 and self.local_rank == 0:
                        warnings.warn(f'the eos_token {repr(token)} is encoded into {tokenized} with length != 1, '
                                      f'using {tokenized[-1]} as the eos_token_id')
                    token = tokenized[-1] # this feature can be changed to eos window if user really want a special word be as eos
                if isinstance(token, int):
                    eos_token_id.append(token)
                elif self.local_rank == 0:
                    warnings.warn(f'the eos_token {repr(token)} is neither str nor int, which is ignored')


        echo = not hide_input

        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
        
        inputs = [i for i in inputs for _ in range(num_return_sequences)]
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in inputs]

        ### Added
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len

        max_length = max_length or self.max_seq_len
        total_len = min(params.max_seq_len, max_new_tokens + max_prompt_len, max_length)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        if output_log_probs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens) +
                                   eos_token_id, dtype=torch.long, device="cuda")

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_pk(probs, top_p, top_k)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if output_log_probs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if output_log_probs:
            token_logprobs = token_logprobs.tolist()

        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_new_tokens]
            probs = None
            if output_log_probs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_new_tokens]
                print(len(probs))
            # cut to after eos tok if any
            for stop_token in list(self.tokenizer.stop_tokens) + eos_token_id:
                try:
                    # mask the input tokens
                    toks_masked = toks.copy()
                    if echo:
                        toks_masked[:len(prompt_tokens[i])] = -100
                    eos_idx = toks_masked.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if output_log_probs else None
                    # print("Find EOS token", stop_token, eos_idx)
                    # print("Tokens", toks)
                    # print("Probs", probs)
                    
                except ValueError:
                    pass
            out_tokens.append(self.tokenizer.decode(toks))
            out_logprobs.append(np.array(probs))

        return GenerateOutput(out_tokens, out_logprobs)

    @torch.no_grad()
    def get_next_token_logits(self, prompt: Union[str, list[str]], candidates: Union[list[str], list[list[str]]], **kwargs) -> list[np.ndarray]:

        if isinstance(prompt, str):
            prompt = [prompt]
        # prompt is a list of strings

        if isinstance(candidates[0], str): 
            candidates = [candidates] * len(prompt)
        # if candidtes is a list of strings, repeat it for each prompt

        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer.encode(cand, bos=False, eos=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[1] if token[0] == 29871 else token[0])  # need to fix it

        bsz = len(prompt)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompts_tokens = [self.tokenizer.encode(p, bos=True, eos=False) for p in prompt]
        max_prompt_size = max(len(t) for t in prompts_tokens)
        tokens = torch.full((bsz, max_prompt_size), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompts_tokens):
            tokens[k, :len(t)] = torch.tensor(t)[:params.max_seq_len].long()

        all_logits = self.model.forward(tokens, 0)[:,-1]
        logits = []
        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())
        return logits

    @torch.no_grad()
    def get_loglikelihood(
            self,
            prefix: str,
            contents: list[str],
    ) -> np.ndarray:

        params = self.model.params
        bsz = len(contents)

        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        prefix_tokens = self.tokenizer.encode(prefix, bos=True, eos=False)
        prompts_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in contents]
        for prompt_tokens in prompts_tokens:
            assert prompt_tokens[: len(prefix_tokens)] == prefix_tokens

        max_prompt_size = max([len(t) for t in prompts_tokens])
        total_len = max_prompt_size
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()

        logits = self.model.forward(tokens[:, :], 0)
        acc_probs = torch.zeros(bsz, dtype=torch.float32).cuda()
        for i in range(len(prefix_tokens), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])

        return acc_probs.cpu().numpy()


def sample_top_pk(probs, p, k):
    """
    Perform top-pk (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.
        k (int): Top-k of tokens to sample.
        
    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-pk sampling selects from the small set of top-k tokens whose cumulative probability mass exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sort = probs_sort[:, :k]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

if __name__ == "__main__":
    llama3_ckpts = "/data/shibo/llama3-ckpts"
    llama_model = Llama3Model(llama3_ckpts, "8B", max_batch_size=3)
    print(llama_model.generate(["Do you love", "The capital of France is"], eos_token_id=[13], output_log_probs=True)) 
    print(llama_model.get_next_token_logits(["The capital of UK is", "The capital of France is", "The capital of Russia is"], ["Paris", "London", "Moscow"]))
    print(llama_model.get_loglikelihood("The capital of UK is", ["The capital of UK is Paris", "The capital of UK is London", "The capital of UK is Moscow"]))

    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 reasoners/lm/llama_3_model.py
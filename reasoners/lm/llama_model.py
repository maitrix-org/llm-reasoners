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
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama1 import ModelArgs, Transformer, Tokenizer

from reasoners import LanguageModel, GenerateOutput


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    return local_rank, world_size


class LlamaModel(LanguageModel):
    def __init__(self, path, size, max_batch_size=1, max_seq_len=2048,
                 local_rank=-1, world_size=-1):
        super().__init__()
        if local_rank == -1 or world_size == -1:
            local_rank, world_size = setup_model_parallel()
        self.tokenizer, self.model = self.load(os.path.join(path, size), os.path.join(path, "tokenizer.model"),
                                               local_rank, world_size, max_batch_size=max_batch_size,
                                               max_seq_len=max_seq_len)
        self.max_seq_len = max_seq_len
        self.local_rank = local_rank

    @staticmethod
    def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int,
             max_seq_len: int) -> Tuple[Tokenizer, Transformer]:
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert (
                world_size == len(checkpoints)
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args).cuda().half()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return tokenizer, model

    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 **kwargs) -> GenerateOutput:
        if max_length is None:
            max_length = self.max_seq_len  # use LLaMA's max length if not set
        if max_new_tokens is None:
            max_new_tokens = max_length  # set to a large number cannot be reached

        if not do_sample:
            if temperature != 0.0 and self.local_rank == 0:  # temperature is explicitly set with do_sample=False #if do_sample=False, temperature set to not 0 have no effect
                warnings.warn('temperature is set, but do_sample=False')
            temperature = 0

        # unify eos_token
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
                    token = tokenized[-1]
                if isinstance(token, int):
                    eos_token_id.append(token)
                elif self.local_rank == 0:
                    warnings.warn(f'the eos_token {repr(token)} is neither str nor int, which is ignored')

        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
        
        inputs = [i for i in inputs for _ in range(num_return_sequences)]
        
        end_pos = torch.zeros(len(inputs)).long().cuda() - 1
        bsz = len(inputs)
        params = self.model.params
        assert bsz <= params.max_batch_size, f"total batch size exceeds limit: {bsz} > {params.max_batch_size}"

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in inputs]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        if max_prompt_size > params.max_seq_len and self.local_rank == 0:
            warnings.warn(f"prompts exceed context length limit: {max_prompt_size} > {params.max_seq_len}")
        total_len = min(params.max_seq_len, max_prompt_size + max_new_tokens)
        total_len = min(total_len, max_length)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()
        input_pos = [len(t) for t in prompt_tokens]
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0

        eos_cnt = torch.zeros(bsz).long().cuda()
        seq_probs = []
        eos_token_id = torch.tensor(eos_token_id, dtype=torch.long).cuda()
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_pk(probs, top_p, top_k)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            seq_probs.append(probs[:, next_token])
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            eos_cnt += torch.any(next_token[:, None] == eos_token_id, dim=-1).long()
            for idx in range(bsz):
                if end_pos[idx].item() == -1:
                    if eos_cnt[idx] > 0 or cur_pos - input_pos[idx] >= max_new_tokens:
                        end_pos[idx] = cur_pos
            if (eos_cnt >= 1).all():
                break

        
        decoded = []
        log_prob = None
        if output_log_probs:
            seq_probs = torch.stack(seq_probs, dim=0)
            log_prob = torch.log(seq_probs)

        for i, (t, input_t) in enumerate(zip(tokens.tolist(), prompt_tokens)):
            t = t[:params.max_seq_len]
            t = t[:len(prompt_tokens[i]) + max_length]#here is the difference, i think you want to set max_new_tokens
            t = [x if x != self.tokenizer.pad_id else self.tokenizer.eos_id for x in t]
            if end_pos[i].item() != -1:
                t = t[:end_pos[i]]
            decoded_tokens = self.tokenizer.decode(t)
            if hide_input:
                decoded_tokens = decoded_tokens[len(inputs[i]):]
            decoded.append(decoded_tokens)

        # TODO: check log_probs
        return GenerateOutput(decoded, log_prob)

    @torch.no_grad()
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)

        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer.encode(cand, bos=False, eos=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[1] if token[0] == 29871 else token[0])

        bsz = len(prompt)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompts_tokens = [self.tokenizer.encode(p, bos=True, eos=False) for p in prompt]
        max_prompt_size = max(len(t) for t in prompts_tokens)
        tokens = torch.full((bsz, max_prompt_size), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompts_tokens):
            tokens[k, :len(t)] = torch.tensor(t)[:params.max_seq_len].long()

        all_logits = self.model.forward(tokens, 0)
        logits = []
        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())
        return logits

    @torch.inference_mode()
    def encode(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.model.tok_embeddings(tokens)
        self.model.freqs_cis = self.model.freqs_cis.to(h.device)
        freqs_cis = self.model.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.model.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.model.norm(h)
        output = self.model.output(h[:, -1, :])  # only compute last logits
        return output.float(), h

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
            assert prompt_tokens[:len(prefix_tokens)] == prefix_tokens

        max_prompt_size = max([len(t) for t in prompts_tokens])
        total_len = max_prompt_size
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t)] = torch.tensor(t)[:params.max_seq_len].long()

        _, h = self.encode(tokens[:, :], 0)
        logits = self.model.output(h)
        acc_probs = torch.zeros(bsz).cuda()
        for i in range(len(prefix_tokens), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])

        return acc_probs.cpu().numpy()

    @staticmethod
    def sample_top_pk(probs, p, k):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sort = probs_sort[:, :k]
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


class DummyLLaMAModel(LanguageModel):
    def __init__(self, path, size, max_batch_size=1, max_seq_len=2048,
                 local_rank=-1, world_size=-1):
        super().__init__()

    @torch.no_grad()
    def generate(
            self,
            inputs: list[str],
            max_gen_len: int = 2048,
            temperature: float = 0.8,
            top_p: float = 0.95,
            end_token: str = "",  # TODO: change this to a function
            hide_input: bool = False,
    ) -> GenerateOutput:
        return GenerateOutput(inputs, np.zeros(len(inputs)))

    @torch.no_grad()
    def get_loglikelihood(
            self,
            prefix: str,
            contents: list[str],
    ) -> np.ndarray:
        return np.zeros(len(contents))

    @torch.no_grad()
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        return [np.zeros(len(cand)) for cand in candidates]


if __name__ == '__main__':
    model = LlamaModel("/data/haotian/RAP_tune/llama-ckpts",'7B')
    print(model.get_next_token_logits(['Smoking is harmful to health.'], candidates=[[' It']]))
    print(model.get_next_token_logits(['Smoking is harmful to health. It'], candidates=[[' is']]))
    print(model.get_next_token_logits(['Smoking is harmful to health. It is'], candidates=[[' the']]))
    print(model.generate(['Smoking is harmful to health.'], max_new_tokens=20, output_log_probs=True, hide_input=False))
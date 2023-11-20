import os
import sys
from typing import Union, Optional
import warnings
import random
import copy
import glob
import time

import torch
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download

from .. import LanguageModel, GenerateOutput


class ExLlamaModel(LanguageModel):
    def __init__(self, model_dir, lora_dir, max_batch_size, max_new_tokens, max_seq_length, device='cuda:0',
                 mem_map: list[int] = None, log_time=False):
        """
        Initializes an ExLlamaModel instance.

        Args:
            model_dir (str): Path to the directory containing the ExLlama model files.
            lora_dir (str): Path to the directory containing the LoRA adapter files (optional).
            device (str): Device to use for inference (e.g. "cpu", "cuda").
            max_batch_size (int): Maximum batch size for inference.
            max_new_tokens (int): Maximum number of new tokens to generate during inference.
            max_seq_length (int): Maximum sequence length for input text.
            mem_map (list[int]): List of integers specifying the memory map for the model (optional).
        Returns:
            None
        """
        try:
            import reasoners
            sys.path.append(os.path.join(os.path.dirname(reasoners.__file__), os.path.pardir, 'exllama'))
            from exllama.generator import ExLlamaGenerator
            from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
            from exllama.tokenizer import ExLlamaTokenizer
            from exllama.lora import ExLlamaLora
        except ImportError as e:
            print(
                '\033[31mError\033[0m: Cannot find exllama submodule. If you clone our repo without "--recursive", running "\033[1mgit submodule update --init\033[0m" under the repo can solve this problem.',
                file=sys.stderr)
            raise e

        if not os.path.isdir(model_dir):
            model_dir = snapshot_download(model_dir)

        super().__init__()
        torch.cuda._lazy_init()

        # set random seed
        torch.set_grad_enabled(False)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        tokenizer_path = os.path.join(model_dir, "tokenizer.model")
        model_config_path = os.path.join(model_dir, "config.json")
        st_pattern = os.path.join(model_dir, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        # Create config, model, tokenizer and generator

        self.config = ExLlamaConfig(model_config_path)  # create config from config.json
        self.config.model_path = model_path  # supply path to model weights file
        self.config.max_seq_len = max_seq_length  # set max sequence length
        self.config.max_input_len = max_seq_length
        self.config.max_attention_size = max_seq_length ** 2
        if mem_map is not None:
            self.config.auto_map = mem_map
        else:
            warnings.warn(
                "mem_map is None, if you want model parallelism, please set mem_map. E.g., to use in 2 * 24G GPUs, set mem_map=[16,22]")

        self.model = ExLlama(self.config)  # create ExLlama instance and load the weights
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)  # create tokenizer from tokenizer model file

        self.cache = ExLlamaCache(self.model, max_batch_size)  # create cache for inference
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)  # create generator
        self.lora = None

        # Load LoRA
        if lora_dir is not None:
            lora_config_path = os.path.join(lora_dir, "adapter_config.json")
            lora_path = os.path.join(lora_dir, "adapter_model.bin")
            self.lora = ExLlamaLora(self.model, lora_config_path, lora_path)
            self.generator.lora = self.lora

        self.device = device
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length

        self.log_time = log_time

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
            eos_token_id: Union[None, str, int, list[str, int]] = None,
            hide_input: bool = True,
            output_log_probs: bool = False,
            **kwargs,
    ) -> GenerateOutput:

        if max_length is not None:
            warnings.warn("max_length is not supported by ExLlamaModel for generation. Use max_new_tokens instead.")
        if max_new_tokens is None:
            warnings.warn(f"max_new_tokens is not set, we will use the default value: {self.max_new_tokens}")
            max_new_tokens = self.max_new_tokens
        if do_sample is False or temperature <= 0.0:
            warnings.warn(
                "do_sample is False while the temperature is non-positive. We will use greedy decoding for Exllama")
            temperature = 1.0
            top_k = 1

        self.generator.settings.token_repetition_penalty_max = 1.0
        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k
        self.generator.settings.typical = 0.0

        eos_token_id_input = copy.deepcopy(eos_token_id)
        eos_token_id = []

        if eos_token_id_input is not None:
            if not isinstance(eos_token_id_input, list):
                eos_token_id_input = [eos_token_id_input]
            for token in eos_token_id_input:
                if isinstance(token, str):
                    tokenized, mask = self.tokenizer.encode(token, return_mask=True, add_bos=False, add_eos=False)
                    tokenized = tokenized[0]
                    if len(tokenized) != 1:
                        warnings.warn(f'the eos_token {repr(token)} is encoded into {tokenized} with length != 1, '
                                      f'using {tokenized[-1]} as the eos_token_id')
                    token = tokenized[-1].item()
                if isinstance(token, int):
                    eos_token_id.append(token)
                else:
                    warnings.warn(f'the eos_token {repr(token)} is neither str nor int, which is ignored')
        eos_token_id.append(self.tokenizer.eos_token_id)

        if num_return_sequences > 1:
            assert len(inputs) == 1, "num_return_sequences > 1 is not supported for batched inputs"
            inputs = inputs * num_return_sequences

        decoded_list = []
        log_prob_list = None
        for start in range(0, len(inputs), self.max_batch_size):
            end = min(start + self.max_batch_size, len(inputs))
            with torch.inference_mode():
                p_time = time.time()
                decoded = self.generate_simple(self.generator, inputs[start:end], max_new_tokens=max_new_tokens,
                                               eos_token_id=eos_token_id, hide_input=hide_input)
                f_time = time.time()
                num_new_tokens = [
                    len(self.tokenizer.encode(d, add_bos=False, add_eos=False)[0])
                    - len(self.tokenizer.encode(e, add_bos=False, add_eos=False)[0])
                    for e, d in zip(inputs[start:end], decoded)]
                t = f_time - p_time
                if self.log_time:
                    tqdm.write(f"Time for generating {sum(num_new_tokens)} tokens: {round(t, 2)}s "
                               f"(speed: {round(sum(num_new_tokens) / t, 2)} t/s)")
            if not isinstance(decoded, list):
                decoded = [decoded]
            # if hide_input:
            #     for i in range(end-start):
            #         decoded[i] = decoded[i][len(inputs[start+i]):]
            #         if isinstance(eos_token_id, str):
            #             decoded[i] = decoded[i].split(eos_token_id)[0]
            log_prob = None
            if output_log_probs:
                warnings.warn(
                    "output_log_probs is temporarily not supported now by ExLlamaModel. Please refere to exllama's code")
            decoded_list.extend(decoded)

        return GenerateOutput(decoded_list, log_prob_list)

    def generate_simple(self, generator, prompt, max_new_tokens=128, eos_token_id=None, hide_input=False):
        # copied from exllama/generator.py
        # support customized eos_token_id

        if eos_token_id is None:
            eos_token_id = [generator.tokenizer.eos_token_id]

        generator.end_beam_search()

        ids, mask = generator.tokenizer.encode(prompt, return_mask=True, max_seq_len=generator.model.config.max_seq_len)
        generator.gen_begin(ids, mask=mask)

        max_new_tokens = min(max_new_tokens, generator.model.config.max_seq_len - ids.shape[1])

        eos = [0] * ids.shape[0]
        early_stop = False
        for i in range(max_new_tokens):
            token = generator.gen_single_token(mask=mask)
            for j in range(token.shape[0]):
                if eos[j] == 0 and token[j, 0].item() in eos_token_id:
                    eos[j] = ids.shape[1] + i
            if all(eos):
                early_stop = True
                break

        sequence = generator.sequence
        for i, eos_pos in enumerate(eos):
            if eos_pos > 0:
                sequence[i, eos_pos + 1:] = generator.tokenizer.pad_token_id
        # if hide_input:
        #     sequence = sequence[:, ids.shape[1]:]
        text = generator.tokenizer.decode(sequence)
        if hide_input:
            text = [text[i][len(prompt[i]):] for i in range(len(prompt))]

        return text

    @torch.no_grad()
    def get_next_token_logits(
            self,
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
                token = self.tokenizer.encode(cand, add_bos=False, add_eos=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[1].item() if len(token) > 1 else token[0].item())

        bsz = len(prompt)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)

        tokens, mask = self.tokenizer.encode(prompt, return_mask=True, add_bos=True, add_eos=False)
        p_time = time.time()
        with torch.no_grad():
            self.cache.current_seq_len = 0
            all_logits = self.model.forward(
                tokens,
                self.cache,
                last_id_only=True,
                preprocess_only=False,
                lora=self.lora,
                output_device=self.device,
                input_mask=mask
            ).squeeze(1)
        assert all_logits.shape[0] == bsz, (all_logits.shape[0], bsz)
        logits = []
        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())
        return logits

    @torch.no_grad()
    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        bsz = len(contents)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)
        prefix_tokens = self.tokenizer.encode(prefix, add_bos=True, add_eos=False).squeeze(0)
        prompts_tokens, mask = self.tokenizer.encode(contents, return_mask=True, add_bos=True, add_eos=False)
        for prompt_tokens in prompts_tokens:
            assert torch.all(prompt_tokens[:len(prefix_tokens)] == prefix_tokens)
        tokens = prompts_tokens
        with torch.no_grad():
            self.cache.current_seq_len = 0
            logits = self.model.forward(
                tokens,
                self.cache,
                last_id_only=False,
                preprocess_only=False,
                lora=self.lora,
                output_device=self.device,
                input_mask=mask
            )
        acc_probs = torch.zeros(bsz).to(self.device)
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])

        return acc_probs.cpu().numpy()

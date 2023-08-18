from exllama import model
from exllama.generator import ExLlamaGenerator
from .. import LanguageModel,GenerateOutput
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.lora import ExLlamaLora
import torch
from typing import Tuple, Union, Optional
import warnings
import numpy as np
import random
import os
from tqdm import tqdm
import glob
import time
class ExLlamaModel(LanguageModel):
    def __init__(self, model_dir, lora_dir, device, max_batch_size, max_new_tokens, max_seq_length):
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

        self.config = ExLlamaConfig(model_config_path)               # create config from config.json
        self.config.model_path = model_path                          # supply path to model weights file
        self.config.max_seq_length = max_seq_length                  # set max sequence length
        self.config.auto_map = [18,22]
        self.model = ExLlama(self.config)                                 # create ExLlama instance and load the weights
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

        self.cache = ExLlamaCache(self.model,max_batch_size)                             # create cache for inference
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)   # create generator
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
            warnings.warn("max_new_tokens is not set, we will use the default value: {}".format(self.max_new_tokens))
            max_new_tokens = self.max_new_tokens
        if eos_token_id is not None:
            warnings.warn("eos_token_id is not supported by ExLlamaModel. Use disallow_tokens instead. Here may lead to bug and please check. We just use split when eos isinstance(str)")
        if do_sample is False:
            warnings.warn("do_sample is defaultly set to False, we will set temp=1.0 and top-k = 1 for Exllama")
            temperature = 1.0
            top_k = 1

        self.generator.settings.token_repetition_penalty_max = 1.0
        self.generator.settings.temperature = temperature
        self.generator.settings.top_p = top_p
        self.generator.settings.top_k = top_k
        self.generator.settings.typical = 0.0

        if num_return_sequences > 1:
            assert len(inputs) == 1, "num_return_sequences > 1 is not supported for batched inputs"
            inputs = inputs * num_return_sequences
        
        decoded_list = []
        log_prob_list = None
        for start in range(0, len(inputs), self.max_batch_size):
            end = min(start + self.max_batch_size, len(inputs))
            with torch.inference_mode():
                p_time = time.time()
                decoded = self.generator.generate_simple(inputs[start:end], max_new_tokens=self.max_new_tokens)
                f_time = time.time()
                print(f"Time for generating {self.max_new_tokens}*{end-start} examples: {f_time-p_time}, speed: {self.max_new_tokens*(end-start)/(f_time-p_time)} t/s")
            if not isinstance(decoded, list):
                decoded = [decoded]
            if hide_input:
                for i in range(end-start):
                    decoded[i] = decoded[i][len(inputs[start+i]):]
                    if isinstance(eos_token_id, str):
                        decoded[i] = decoded[i].split(eos_token_id)[0]
            log_prob = None
            if output_log_probs:
                warnings.warn("output_log_probs is temporarily not supported by ExLlamaModel. Please refere to exllama's code")
            decoded_list.extend(decoded)
        return GenerateOutput(decoded_list, log_prob_list)

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
                last_id_only = True,
                preprocess_only = False,
                lora = self.lora,
                output_device = self.device,
                input_mask = mask
            ).squeeze(1)
        f_time = time.time()
        assert all_logits.shape[0] == bsz, (all_logits.shape[0], bsz)
        print(f"Time for forwarding {self.max_new_tokens} examples: {f_time-p_time}, speed: {self.max_new_tokens/(f_time-p_time)} t/s")
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
                last_id_only = False,
                preprocess_only = False,
                lora = self.lora,
                output_device = self.device,
                input_mask = mask
            )
        acc_probs = torch.zeros(bsz).to(self.device)
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        return acc_probs.cpu().numpy()
        
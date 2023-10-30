import os
import warnings
from typing import Union, Optional

import scipy
import numpy as np
from transformers import StoppingCriteriaList

from reasoners import LanguageModel, GenerateOutput


class LlamaCppModel(LanguageModel):
    def __init__(self, path, n_ctx=2048, n_batch=512, n_thread=None):

        try:
            from llama_cpp import Llama
        except ImportError as e:
            Llama = None
            raise ImportError('You need to install llama-cpp-python if you want to use llama_cpp. '
                              'To ensure CUDA enabled, run \033[1mLLAMA_CUBLAS=1 CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose\033[0m. '
                              'If your build fails or need more details, visit https://github.com/Ber666/llm-reasoners/tree/main/reasoners/lm#llamacpp. '
                              'If you want to use facebookresearch/llama, use llama instead of llama_cpp.') from e

        self.llama = Llama(path,
                           n_ctx=n_ctx,
                           n_batch=n_batch,
                           n_gpu_layers=-1,  # set to a large number to put all layers on GPUs
                           logits_all=True,
                           verbose=False,
                           n_threads=n_thread)
        self.n_ctx = n_ctx

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
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **kwargs) -> GenerateOutput:
        assert hide_input, 'TODO: only supports hide_input=True for llama.cpp now'
        assert max_length is None, 'TODO: does not support max_length for llama.cpp now'
        assert num_return_sequences == 1, 'TODO: does not support multiple return sequences for llama.cpp now'
        if isinstance(eos_token_id, int):
            assert False, 'TODO: does not support *int* eos_token_id for llama.cpp now, use *str*'
        if isinstance(eos_token_id, list):
            assert not any(isinstance(token, int) for token in
                           eos_token_id), 'TODO: does not support *int* eos_token_id for llama.cpp now, use *str*'
        if not do_sample:
            if temperature != 1.0:  # temperature is explicitly set with do_sample=False
                warnings.warn('temperature is set, but do_sample=False')
            temperature = 0

        generated_text = []
        log_probs = [] if output_log_probs else None

        for input in inputs:
            if max_new_tokens is None:
                max_new_tokens = self.n_ctx
            output = self.llama(input, max_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                                stop=eos_token_id, logprobs=1 if output_log_probs else None,
                                stopping_criteria=stopping_criteria)['choices'][0]
            generated_text.append(output['text'])
            if output_log_probs:
                token_logprobs = output['logprobs']['token_logprobs']
                log_probs.append(np.array(token_logprobs))
        
        return GenerateOutput(text=generated_text, log_prob=log_probs)

    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,  # log_softmax, softmax, TODO: need docstring
                              **kwargs) -> list[np.ndarray]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)

        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenize(cand, add_bos=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[0])

        output = []

        for input, cand in zip(prompt, cand_tokens):
            self.llama.reset()
            input_tokens = self.tokenize(input, add_bos=True)
            self.llama.eval(input_tokens)
            logits = np.array(self.llama.eval_logits[-1])
            if postprocess == 'log_softmax':
                logits = scipy.special.log_softmax(logits, axis=-1)
            elif postprocess == 'softmax':
                logits = scipy.special.softmax(logits, axis=-1)
            output.append(logits[cand])
        return output

    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        prefix_tokens = self.tokenize(prefix, add_bos=True)
        output = []
        for content in contents:
            content_tokens = self.tokenize(content, add_bos=True)
            if any(p != c for p, c in zip(prefix_tokens, content_tokens)):
                warnings.warn(f'prefix {repr(prefix)} does not match content {repr(content)}')
            self.llama.reset()
            self.llama.eval(content_tokens)
            logits = self.llama.eval_logits
            log_probs = scipy.special.log_softmax(logits, axis=-1)
            content_log_probs = log_probs[np.arange(len(prefix_tokens), len(content_tokens)) - 1,
            content_tokens[len(prefix_tokens):]]
            output.append(sum(content_log_probs))
        return np.array(output)

    def tokenize(self, text: str, add_bos=True):
        return self.llama.tokenize(bytes(text, encoding='utf-8'), add_bos=add_bos)


if __name__ == '__main__':
    from timeit import timeit
    model = LlamaCppModel(path='/data/yi/llama.cpp/models/65B/ggml-model-q5_0.bin')
    print(model.get_next_token_logits(['Hello'], candidates=[[',']], postprocess='log_softmax'))
    print(model.get_next_token_logits(['Hello,'], candidates=[[' I']], postprocess='log_softmax'))
    print(model.get_next_token_logits(['Hello, I'], candidates=[[' am']], postprocess='log_softmax'))
    print(model.generate(['Hello'], max_new_tokens=20, output_log_probs=True))

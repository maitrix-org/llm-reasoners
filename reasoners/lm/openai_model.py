import os
import warnings

import openai
import numpy as np
from typing import Optional, Union, Literal
import time

from .. import LanguageModel, GenerateOutput


class OpenAIModel(LanguageModel):
    def __init__(self, model: str, rate_limit_in_seconds: float = 2,
                 api_type: Optional[Literal['completions', 'chat']] = None, max_retries: int = 20):
        self.model = model
        self.rate_limit_in_seconds = rate_limit_in_seconds
        self.last_query_time = 0
        self.max_retries = max_retries
        if api_type is None:
            if 'gpt-4' in model or 'gpt-3.5' in model:
                api_type = 'chat'
            else:
                api_type = 'completions'
        self.api_type = api_type
        openai.api_key = os.environ.get("OPENAI_API_KEY", None)
        if openai.api_key is None:
            raise ValueError("OPENAI_API_KEY not found in env. Use `export OPENAI_API_KEY=<your key>` to set it.")

    def generate(self,
                 inputs: list[str],
                 max_length: None = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: None = None,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, list[str]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: None = None,
                 **kwargs) -> GenerateOutput:
        if not do_sample:
            if temperature != 1.0:  # temperature is explicitly set with do_sample=False
                warnings.warn('temperature is set, but do_sample=False')
            temperature = 0
        if stopping_criteria is not None:
            warnings.warn('OpenAI model does not support stopping_criteria, ignoring it')

        eos_token_id_input = eos_token_id
        eos_token = []
        if eos_token_id_input is not None:
            if not isinstance(eos_token_id_input, list):
                eos_token_id_input = [eos_token_id_input]
            for token in eos_token_id_input:
                if isinstance(token, str):
                    eos_token.append(token)
                else:
                    warnings.warn(f'the eos_token {repr(token)} is neither str, which is ignored')

        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'

        if top_k is not None:
            warnings.warn('OpenAI model does not support top_k, ignoring it')

        if output_log_probs and self.api_type == 'chat':
            warnings.warn('output_log_probs is not supported for ChatCompletion, ignoring it')
            output_log_probs = False

        result_text = []
        if output_log_probs:
            result_log_prob = []
        else:
            result_log_prob = None

        for inp in inputs:
            text, log_prob = self.query_openai(prompt=inp,
                                               max_tokens=max_new_tokens,
                                               temperature=temperature,
                                               top_p=top_p,
                                               num_return_sequences=num_return_sequences,
                                               stop=eos_token,
                                               **kwargs)
            result_text += text
            if output_log_probs:
                result_log_prob += log_prob

        return GenerateOutput(text=result_text, log_prob=None)

    def query_openai(self,
                     prompt: str,
                     max_tokens: int = None,
                     temperature: float = 1.0,
                     top_p: float = 1.0,
                     num_return_sequences: int = 1,
                     stop: Optional[list[str]] = None,
                     **kwargs) -> GenerateOutput:
        for retry in range(self.max_retries):
            try:
                if time.time() - self.last_query_time < self.rate_limit_in_seconds:
                    time.sleep(self.rate_limit_in_seconds - (time.time() - self.last_query_time))
                if self.api_type == 'chat':
                    messages = [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}]
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        **kwargs
                    )
                    return GenerateOutput(text=[choice["message"]["content"] for choice in response["choices"]])
                elif self.api_type == 'completions':
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=0,
                        **kwargs
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response["choices"]],
                        log_prob=[choice["logprobs"] for choice in response["choices"]]
                    )
            except openai.error.RateLimitError:
                warnings.warn('RateLimitError from openai, waiting for 5 seconds')
                time.sleep(5)
                pass
            except openai.error.APIError as e:
                if 'Bad gateway' in e:
                    warnings.warn('Bad Gateway from openai, waiting for 5 seconds')
                    time.sleep(5)
                    pass
                raise e
            #  let other exceptions raise
        # exceed max_retries
        raise openai.error.OpenAIError

    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                          prompt: Union[str, list[str]],
                          **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")


if __name__ == '__main__':
    model = OpenAIModel(model='gpt-3.5-turbo')
    print(model.generate(['Hello, how are you?', 'How to go to Shanghai from Beijing?']))

import os
import openai
import numpy as np
from typing import Optional, Union, Literal
import time

from reasoners.base import LanguageModel, GenerateOutput
from openai import OpenAI

import pickle

PROMPT_TEMPLATE_ANSWER = 'Your response need to be ended with "So the answer is"\n\n'
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"


class DeepseekModel(LanguageModel):
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature=0.0,
        additional_prompt=None,
        is_instruct_model: bool = False,
        task_dir: str = None
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt
        self.is_instruct_model = is_instruct_model
        self.task_dir = task_dir
        self.__init_client__()

    def __init_client__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", None),
        )

    def generate(
        self,
        prompt: Optional[Union[str, list[str]]],
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=64,
        **kwargs,
    ) -> GenerateOutput:

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs

        if isinstance(prompt, list):
            # @zj: why can't we pass a list of prompts?
            assert len(prompt) == 1
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                messages = [{"role": "user", "content": prompt}]
                # print(messages)
                # print("SAnity check")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop,
                )

                # save response pickle object
                utc_timestamp = int(time.time())
                response_path = os.path.join(
                    self.task_dir, f"{utc_timestamp}.pkl")
                with open(response_path, "wb") as f:
                    pickle.dump(response, f)

                return GenerateOutput(
                    text=[choice.message.content for choice in response.choices],
                    log_prob=None,
                )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        # after 64 tries, still no luck
        raise RuntimeError(
            "GPTCompletionModel failed to generate output, even after 64 tries"
        )

    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]],
        **kwargs,
    ) -> list[np.ndarray]:
        raise NotImplementedError(
            "GPTCompletionModel does not support get_next_token_logits"
        )

    def get_loglikelihood(
        self, prompt: Union[str, list[str]], **kwargs
    ) -> list[np.ndarray]:
        raise NotImplementedError(
            "GPTCompletionModel does not support get_log_prob")

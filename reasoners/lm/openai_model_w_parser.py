import os
import numpy as np
from typing import Optional, Union, Literal, Callable, Tuple
import time
import logging

from openai import OpenAI

from reasoners.base import LanguageModel, GenerateOutput

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ANSWER = 'Your response need to be ended with "So the answer is"\n\n'
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"


class OpenAIModel(LanguageModel):
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature=0.0,
        additional_prompt=None,
        backend: Literal["openai", "sglang"] = "openai",
        is_instruct_model: bool = False,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend = backend
        self.additional_prompt = additional_prompt
        self.is_instruct_model = is_instruct_model
        self.__init_client__()

    def __init_client__(self):
        if self.backend == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", None),
            )
        elif self.backend == "sglang":
            self.client = OpenAI(
                base_url=os.getenv("SGLANG_API_URL", None),
            )
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(
        self,
        prompt: Optional[Union[str, list[str]]],
        system_prompt: Optional[Union[str, list[str]]] = None,
        base64_image: Optional[str] = None,
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=1,
        parser: Callable[[str], Tuple[str, bool, Optional[str]]] = lambda x: (
            x,
            True,
            None,
        ),
        **kwargs,
    ) -> GenerateOutput:

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs
        num_return_sequences = kwargs.pop("n", num_return_sequences)

        if isinstance(prompt, list):
            assert len(prompt) == 1  # @zj: why can't we pass a list of prompts?
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        is_instruct_model = self.is_instruct_model
        if not is_instruct_model:
            # Recheck if the model is an instruct model with model name
            model_name = self.model.lower()
            if (
                ("gpt-3.5" in model_name)
                or ("gpt-4" in model_name)
                or ("instruct" in model_name)
            ):
                is_instruct_model = True

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                if is_instruct_model:
                    messages = (
                        [
                            {"role": "system", "content": system_prompt},
                        ]
                        if system_prompt is not None
                        else []
                    )
                    if base64_image is not None:
                        user_msg = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": base64_image},
                                },
                            ],
                        }
                    else:
                        user_msg = {"role": "user", "content": prompt}

                    messages.append(user_msg)

                    logger.debug(f"OpenAIModel.generate() `messages`: \n{messages}")
                    logger.debug(f"kwargs: {kwargs}")
                    logger.debug(f"max_tokens: {max_tokens}")

                    # Calculate the size of the payload in bytes
                    # payload = {
                    #     "model": self.model,
                    #     "messages": messages,
                    #     "max_tokens": max_tokens,
                    #     "temperature": temperature,
                    #     "top_p": top_p,
                    #     "n": num_return_sequences,
                    #     "stop": stop,
                    #     **kwargs,
                    # }
                    # import json

                    # payload_json = json.dumps(payload)
                    # payload_size = len(payload_json.encode("utf-8"))

                    # logger.debug(f"Payload size: {payload_size} bytes")

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        # stop=stop,
                        n=num_return_sequences,
                        **kwargs,
                    )

                    logger.debug(f"OpenAIModel.generate() `response`: \n{response}")

                    return GenerateOutput(
                        text=[
                            parser(choice.message.content)[0]
                            for choice in response.choices
                        ],
                        log_prob=[choice.logprobs for choice in response.choices],
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        # stop=stop,
                        logprobs=0,
                        **kwargs,
                    )
                    return GenerateOutput(
                        text=[parser(choice["text"])[0] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]],
                    )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        # after 64 tries, still no luck
        raise RuntimeError(
            f"GPTCompletionModel failed to generate output, even after {retry} tries"
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
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")


if __name__ == "__main__":
    model = OpenAIModel(model="gpt-3.5-turbo")
    print("-------OpenAI client-------")
    print(model.generate(["How to go to Shanghai from Beijing?"]))
    print("-------SGLang client-------")
    model = OpenAIModel(
        model="meta-llama/Llama-3.1-8B-Instruct",
        backend="sglang",
        is_instruct_model=True,
    )
    print(model.generate(["How to go to Shanghai from Beijing?"]))

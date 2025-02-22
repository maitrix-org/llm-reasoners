import os
import numpy as np
from typing import Optional, Union, Literal
import time

from reasoners.base import LanguageModel, GenerateOutput
from openai import OpenAI
import pickle
from datetime import datetime


class OpenAIModel(LanguageModel):
    def __init__(
        self,
        backend: Literal["openai", "sglang"],
        model: str,
        task_dir: str
    ):
        self.backend = backend
        self.model = model
        self.task_dir = task_dir
        self.__init_client__()

    def log(self, text: str):
        current_time = datetime.now()
        formatted_time = current_time.strftime("[%Y%m%d] - %H:%M.%S")
        if text.startswith("\n"):
            text = f"\n{formatted_time}\n{text.lstrip()}"
        print(text)
        with open(f"{self.task_dir}/log.txt", "a+") as f:
            f.write(f"{text}\n")

    def __init_client__(self):
        if self.backend == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", None),
            )
        elif self.backend == "sglang":
            self.client = OpenAI(
                base_url="http://127.0.0.1:30000/v1",
                api_key="None"
            )
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

    def generate(
        self,
        prompt: Optional[Union[str, list[str]]],
        num_return_sequences: int = 1,
        temperature: float = 0.6,
        max_new_tokens: int = 8196,
        n_retry=4,
        **kwargs,
    ) -> GenerateOutput:
        self.log("llm_generate()")
        for i in range(1, n_retry + 1):
            try:
                messages = [{"role": "user", "content": prompt}]
                if "deepseek-r1" in self.model.lower():
                    messages.append({"role": "assistant", "content": "<think>\n"})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=num_return_sequences,
                    **kwargs,
                )

                utc_timestamp = int(time.time())
                # save prompt
                with open(os.path.join(self.task_dir, f"{utc_timestamp}-prompt.txt"), "w+") as f:
                    for message in messages:
                        f.write(message["content"])
                # save responses
                for idx, choice in enumerate(response.choices):
                    with open(os.path.join(self.task_dir, f"{utc_timestamp}-response-{idx+1}.txt"), "w+") as f:
                        f.write(choice.message.content)
                # save response pickle object
                with open(os.path.join(self.task_dir, f"{utc_timestamp}.pkl"), "wb") as f:
                    pickle.dump(response, f)


                return GenerateOutput(
                    text=[choice.message.content for choice in response.choices],
                    log_prob=None,
                )

            except Exception as e:
                self.log(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        # after 64 tries, still no luck
        raise RuntimeError(
            f"GPTCompletionModel failed to generate output, even after {n_retry} tries"
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
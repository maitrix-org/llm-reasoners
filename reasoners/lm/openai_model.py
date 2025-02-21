import os
import numpy as np
from typing import Optional, Union, Literal
import time

from reasoners.base import LanguageModel, GenerateOutput
from openai import OpenAI
import pickle


class OpenAIModel(LanguageModel):
    def __init__(
        self,
        model: str,
        # max_tokens: int = 8192,
        max_tokens: int = 2048,
        temperature=0.0,
        additional_prompt=None,
        backend: Literal["openai", "sglang"] = "openai",
        is_instruct_model: bool = False,
        task_dir: str = None
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend = backend
        self.additional_prompt = additional_prompt
        self.is_instruct_model = is_instruct_model
        self.task_dir = task_dir
        self.__init_client__()

    def log(self, text: str):
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
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        temperature=None,
        retry=64,
        **kwargs,
    ) -> GenerateOutput:
        self.log("llm_generate()")
        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "<think>\n"}]
                if "deepseek-r1" in self.model.lower():
                    messages.append({"role": "assistant", "content": "<think>\n"})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop,
                    **kwargs,
                )

                utc_timestamp = int(time.time())
                # save prompt
                with open(os.path.join(self.task_dir, f"{utc_timestamp}-prompt.txt"), "w+") as f:
                    f.write(prompt)
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

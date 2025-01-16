import os
import openai
import numpy as np
from typing import Optional, Union, Literal
import time
import requests
from reasoners.base import LanguageModel, GenerateOutput
from openai import OpenAI
import warnings
from transformers import AutoTokenizer

PROMPT_TEMPLATE_ANSWER = 'Your response need to be ended with "So the answer is"\n\n'
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

class SGLangModel(LanguageModel):
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 2048,
        temperature=0.0,
        additional_prompt=None,
        is_instruct_model: bool = False,
    ):
        try:
            import sglang as sgl
        except ImportError:
            raise ImportError("Please install sglang package to use SGLangModel")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt
        self.is_instruct_model = is_instruct_model
        self.tokenizer = AutoTokenizer.from_pretrained(model, lagacy=False, trust_remote_code=True)
        self.__init_client__()

    def __init_client__(self):
        self.client = OpenAI(
            base_url=os.getenv("SGLANG_API_URL", None),
        )

    def generate(
        self,
        prompt: Optional[Union[str, list[str]]],
        max_length: Optional[int] = None,
        max_new_tokens: int = None,
        top_k: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=1,
        hide_input=True,
        do_sample: bool = False,
        **kwargs,
    ) -> GenerateOutput:
        if not hide_input:
            raise ValueError("hide_input must be True for SGLangModel")

        if max_length is not None:
            warnings.warn("max_length is not supported by SGLangModel for generation. Use max_new_tokens instead.")

        if top_k is not None:
            warnings.warn("top_k is not supported by SGLangModel for generation. Use top_p (nucleus sampling) instead.")
        
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs
        
        if not do_sample or temperature == 0.0:
            warnings.warn('temperature=0.0 is equivalent to greedy search, ')
            do_sample = False
            temperature = 1.0

        if eos_token_id is not None:
            if isinstance(eos_token_id, str):
                eos_token_id = [eos_token_id]
            # the eos_token_id is for the compatibility with the other models
            assert stop is None and isinstance(eos_token_id, list) and all(isinstance(e, str) for e in eos_token_id), \
                "eos_token_id should be a string or list of strings for SGLangModel"
            stop = eos_token_id

        if isinstance(prompt, list):
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

        is_instruct_model = self.is_instruct_model
        if not is_instruct_model:
            # Recheck if the model is an instruct model with model name
            model_name = self.model.lower()
            if ("instruct" in model_name):
                print(f"Warning: The model name '{self.model}' contains 'instruct', but is_instruct_model is set to False.")

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                if is_instruct_model:
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=logprobs,
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=[token.logprob for token in response.choices[0].logprobs.content] if logprobs else None,
                    )
                else:
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=logprobs,
                        **kwargs,
                    )
                    return GenerateOutput(
                        text=[choice.text for choice in response.choices],
                        log_prob=[choice.logprobs.token_logprobs for choice in response.choices] if logprobs else None,
                    )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        # after 64 tries, still no luck
        raise RuntimeError(
            "CompletionModel failed to generate output, even after 64 tries"
        )

    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]],
        **kwargs,
    ) -> list[np.ndarray]:
        raise NotImplementedError(
            "CompletionModel does not support get_next_token_logits"
        )

    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        
        import sglang as sgl
        from sglang.api import set_default_backend
        from sglang import RuntimeEndpoint
        
        actions = []
        for c in contents:
            if c.startswith(prefix):
                action = c[len(prefix):].strip()  # Remove the prefix and strip spaces
                actions.append(action)
            else:
                raise ValueError(f"'{prefix}' is not a prefix of '{c}'")
        
        base_url=os.getenv("SGLANG_API_URL", None)
        url = base_url.split("/", 3)[:3]
        url = "/".join(url)
        set_default_backend(RuntimeEndpoint(url))

        @sgl.function
        def helper(s):
            s += prefix + sgl.gen("logprob", choices=actions)

        state = helper.run()
        meta_info = state.get_meta_info("logprob")
        return np.array(meta_info['normalized_prompt_logprobs'])
    
    

if __name__ == "__main__":
    model = OpenAIModel(
        model="meta-llama/Llama-3.1-8B-Instruct",
        is_instruct_model=True,
    )
    print(model.generate(["How to go to Shanghai from Beijing?"]))
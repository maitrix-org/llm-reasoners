import os
from click import Option
import openai
import numpy as np
from typing import Optional, Union
import time

from .. import LanguageModel, GenerateOutput
from openai import OpenAI
class GPTCompletionModel(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 2048, temperature=0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(
            api_key = os.getenv("YI_PERSONAL", None),
            organization='org-vEqqgGhqCUXNYuFXPIeVg2ME',
        )
    
    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 20,
                stop: Optional[str] = None,
                logprobs: Optional[int] = None,
                temperature = None,
                **kwargs) -> GenerateOutput:
        
        gpt_temperature = self.temperature if temperature is None else temperature
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        prompt = "Your response need to be ended with \"So the answer is\"\n\n" + prompt
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        if logprobs is None:
            logprobs = 0

        i = 1

        for i in range(1, 65):  # try 64 times
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if ('gpt-3.5' in self.model) or ('gpt-4' in self.model):
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=gpt_temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop
                    )
                    '''print('-----------------------------------------')
                    print(f'Prompt:\n{prompt}')
                    print('-------------prompt end------------------')
                    print('-----------------------------------------')
                    print('Response:')
                    for i, choice in enumerate(response["choices"]):
                        print(f'---------response {i}------------')
                        print(choice["message"]["content"])
                    print('-------------response end----------------') '''

                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=gpt_temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=logprobs,
                        **kwargs
                    )

                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]]
                    )
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i*10} seconds")
                time.sleep(i*10)
        
        # after 64 tries, still no luck
        raise RuntimeError("GPTCompletionModel failed to generate output, even after 64 tries")
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("GPTCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                    prompt: Union[str, list[str]],
                    **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")
import os

import numpy as np
from typing import Optional, Union
import time
from .. import LanguageModel, GenerateOutput
import anthropic


PROMPT_TEMPLATE_ANSWER = "Your response need to be ended with \"So the answer is\"\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

class ClaudeModel(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 1024, temperature=0.0, additional_prompt=None):
        self.client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt
    
    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                temperature = None,
                additional_prompt=None,
                retry = 64,
                **kwargs) -> GenerateOutput:
        
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        

        temperature = self.temperature if temperature is None else temperature

        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")

        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        if max_tokens is None:
            max_tokens = self.max_tokens

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                text = message.content[0].text

                return GenerateOutput(
                    text=[text],
                    log_prob=None
                )
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i^2)
                #request exceed is common, so set square sleep time
        raise RuntimeError("ClaudeModel failed to generate output, even after 64 tries")
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("ClaudeModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                    prompt: Union[str, list[str]],
                    **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("ClaudeModel does not support get_log_prob")
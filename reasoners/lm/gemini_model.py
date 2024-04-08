import os

import numpy as np
from typing import Optional, Union
import time
import google.generativeai as genai
from .. import LanguageModel, GenerateOutput
from google.generativeai.types import HarmCategory, HarmBlockThreshold
GEMINI_KEY = os.getenv("GEMINI_KEY", None)#user need to set the environment variable GEMINI_KEY
genai.configure(api_key=GEMINI_KEY)

PROMPT_TEMPLATE_ANSWER = "Your response need to be ended with \"So the answer is\"\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"
class BardCompletionModel(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 2048, temperature=0.0, additional_prompt=None):
        self.model = genai.GenerativeModel(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_prompt = additional_prompt
    
    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                rate_limit_per_min: Optional[int] = 60,
                additional_prompt=None,
                retry = 64,
                **kwargs) -> GenerateOutput:
        
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

        if max_tokens is None:
            max_tokens = self.max_tokens

        prompt = [prompt]
        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                
                messages = [{"role": "user", "parts": prompt}]
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                }
                response = self.model.generate_content(
                    contents=messages,
                    safety_settings=safety_settings,
                )

                return GenerateOutput(
                    text=[response.text],
                    log_prob=None
                )
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        raise RuntimeError("BardCompletionModel failed to generate output, even after 64 tries")
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("BardCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                    prompt: Union[str, list[str]],
                    **kwargs) -> list[np.ndarray]:
        
        raise NotImplementedError("BardCompletionModel does not support get_log_prob")
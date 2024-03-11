import os

import numpy as np
from typing import Optional, Union
import time
import google.generativeai as genai
from .. import LanguageModel, GenerateOutput
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GEMINI_KEY = os.getenv("GEMINI_KEY", None)
genai.configure(api_key=GEMINI_KEY)


class BardCompletionModel(LanguageModel):
    def __init__(self, model: str, max_tokens: int = 2048, temperature=0.0):
        self.model = genai.GenerativeModel(model)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self,
                 prompt: Optional[Union[str, list[str]]],
                 max_tokens: int = None,
                 top_p: float = 1.0,
                 rate_limit_per_min: Optional[int] = 60,
                 temperature=None,
                 **kwargs) -> GenerateOutput:

        gpt_temperature = self.temperature if temperature is None else temperature
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        prompt = "Your response need to be ended with \"So the answer is\"\n\n" + prompt
        if max_tokens is None:
            max_tokens = self.max_tokens

        i = 1
        prompt = [prompt]
        for i in range(1, 65):  # try 64 times
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                config = genai.GenerationConfig(
                    temperature=gpt_temperature,
                )
                messages = [{"role": "user", "parts": prompt}]
                # safety_settings={'HARASSMENT':'BLOCK_NONE','HATE_SPEECH':'BLOCK_NONE','DANGEROUS_CONTENT':'BLOCK_NONE','SEXUALLY_EXPLICIT':'BLOCK_NONE'}
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                }
                response = self.model.generate_content(
                    contents=messages,
                    safety_settings=safety_settings,
                    generation_config=config,
                )
                # print(response.prompt_feedback)
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
                    text=[response.text],
                    log_prob=None
                )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i * 10} seconds")
                time.sleep(i * 10)

        # after 64 tries, still no luck
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
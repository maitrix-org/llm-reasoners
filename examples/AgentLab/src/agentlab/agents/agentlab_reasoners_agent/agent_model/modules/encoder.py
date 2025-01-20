from abc import abstractmethod

from ..base import AgentModule
import json


class BaseEncoder(AgentModule):
    def __init__(self, identity, *args, **kwargs):
        self.identity = identity

    @abstractmethod
    def __call__(self, observation, **kwargs): ...


class PromptedEncoder(BaseEncoder):
    def __init__(self, identity, llm, prompt_template, parser):
        super().__init__(identity)
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template
        self.parser = parser

    def __call__(self, observation_text, observation_screenshot, memory, verbose=False, **kwargs):
        user_prompt = self.prompt_template.format(
            observation=observation_text, memory=memory, **kwargs
        )

        if verbose:
            print("===========================PromptedEncoder===========================")
            print("SYSTEM PROMPT in PromptedEncoder:")
            print(str(self.identity))
            print("-" * 100)
            print(f"USER PROMPT in PromptedEncoder:")
            print(user_prompt)

        llm_outputs = self.llm(
            # system_prompt=str(self.identity), user_prompt=user_prompt, parser=self.parser, **kwargs
            system_prompt=str(self.identity),
            user_prompt=user_prompt,
            base64_image=observation_screenshot,
            parser=self.parser,
            **kwargs,
        )

        return llm_outputs[0]

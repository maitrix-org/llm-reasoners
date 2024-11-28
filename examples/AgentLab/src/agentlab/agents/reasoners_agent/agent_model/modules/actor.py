from ..base import AgentModule
import json


class BaseActor(AgentModule):
    ...


class PromptedActor(BaseActor):
    def __init__(self, identity, llm, prompt_template, parser):
        super().__init__()
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template
        self.parser = parser

    def __call__(self, observation, observation_screenshot, state, memory, intent, **kwargs):

        user_prompt = self.prompt_template.format(
            observation=observation, state=state, memory=memory, intent=intent, **kwargs
        )
        llm_outputs = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, base64_image=observation_screenshot, parser=self.parser, **kwargs
        )

        return llm_outputs[0]

from abc import abstractmethod

from ..base import AgentModule


class Encoder(AgentModule):
    @abstractmethod
    def __call__(self, observation, *args, **kwargs): ...


class PromptedEncoder(Encoder):
    def __init__(self, identity, llm, prompt_template):
        super().__init__()
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template

    def __call__(self, observation, memory, **kwargs):
        user_prompt = self.prompt_template.format(
            observation=observation, memory=memory, **kwargs
        )
        llm_output = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **kwargs
        )

        return llm_output
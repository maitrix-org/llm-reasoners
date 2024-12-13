from abc import abstractmethod
from ..base import AgentModule


class Critic(AgentModule):
    @abstractmethod
    def __call__(self, state, memory, *args, **kwargs): ...


class PromptedCritic(Critic):
    def __init__(self, identity, llm, prompt_template):
        super().__init__()
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template

    def __call__(self, state, memory, llm_kwargs=None, **kwargs):
        if llm_kwargs is None:
            llm_kwargs = {}
        user_prompt = self.prompt_template.format(state=state, memory=memory, **kwargs)
        llm_output = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **llm_kwargs
        )

        return llm_output

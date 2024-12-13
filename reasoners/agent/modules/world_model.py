from abc import abstractmethod
from ..base import AgentModule


class WorldModel(AgentModule):
    @abstractmethod
    def __call__(self, state, memory, plan, **kwargs): ...


class PromptedWorldModel(WorldModel):
    def __init__(self, identity, llm, prompt_template):
        super().__init__()
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template

    def __call__(self, state, memory, plan, llm_kwargs=None, **kwargs):
        if llm_kwargs is None:
            llm_kwargs = {}
        user_prompt = self.prompt_template.format(
            state=state, memory=memory, plan=plan, **kwargs
        )
        llm_output = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **llm_kwargs
        )

        return llm_output
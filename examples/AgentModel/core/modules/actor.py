from ..base import AgentModule


class BaseActor(AgentModule): ...


class PromptedActor(BaseActor):
    def __init__(self, identity, llm, prompt_template):
        super().__init__()
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template

    def __call__(self, observation, state, memory, intent, llm_kwargs=None, **kwargs):
        if llm_kwargs is None:
            llm_kwargs = {}
            
        user_prompt = self.prompt_template.format(
            observation=observation, state=state, memory=memory, intent=intent, **kwargs
        )
        llm_response = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **llm_kwargs
        )
        return llm_response

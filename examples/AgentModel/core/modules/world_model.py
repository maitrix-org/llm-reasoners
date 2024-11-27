from ..base import AgentModule


class BaseWorldModel(AgentModule):
    def __init__(self, identity):
        self.identity = identity

    def __call__(self, state, memory, intent, **kwargs):
        raise NotImplementedError


class PromptedWorldModel(BaseWorldModel):
    def __init__(self, identity, llm, prompt_template):
        super().__init__(identity)
        self.llm = llm
        self.prompt_template = prompt_template

    def __call__(self, state, memory, intent, llm_kwargs=None, **kwargs):
        if llm_kwargs is None:
            llm_kwargs = {}
        user_prompt = self.prompt_template.format(
            state=state, memory=memory, intent=intent, **kwargs
        )
        llm_output = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **llm_kwargs
        )

        return llm_output
    
class KnowledgePromptedWorldModel(PromptedWorldModel): 
    def __init__(self, identity, llm, prompt_template, knowledge):
        super().__init__(identity, llm, prompt_template)
        self.knowledge = knowledge
        
    def __call__(self, state, memory, intent, llm_kwargs=None, **kwargs):
        return super().__call__(state, memory, intent, knowledge=self.knowledge, llm_kwargs=llm_kwargs, **kwargs)
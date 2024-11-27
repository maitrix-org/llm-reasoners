from abc import abstractmethod

from ..base import AgentModule


class BaseEncoder(AgentModule):
    def __init__(self, identity, *args, **kwargs):
        self.identity = identity

    @abstractmethod
    def __call__(self, observation, **kwargs): ...


class PromptedEncoder(BaseEncoder):
    def __init__(self, identity, llm, prompt_template):
        super().__init__(identity)
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

class StateMemoryUpdateEncoder(BaseEncoder):
    def __init__(self, identity, 
                 state_llm, state_prompt_template, 
                 memory_update_llm, memory_update_prompt_template):
        super().__init__(identity)
        self.identity = identity
        self.state_llm = state_llm
        self.state_prompt_template = state_prompt_template
        self.memory_update_llm = memory_update_llm
        self.memory_update_prompt_template = memory_update_prompt_template

    def __call__(self, observation, memory, **kwargs): 
        state_prompt = self.state_prompt_template.format(
            observation=observation, memory=memory, **kwargs
        )
        state_llm_output = self.state_llm(
            system_prompt=str(self.identity), user_prompt=state_prompt, **kwargs
        )
        state = state_llm_output['state']
        
        memory_update_prompt = self.memory_update_prompt_template.format(
            observation=observation, memory=memory, state=state, **kwargs
        )
        memory_update_llm_output = self.memory_update_llm(
            system_prompt=str(self.identity), user_prompt=memory_update_prompt, **kwargs
        )
        memory_update = memory_update_llm_output['memory_update']
        
        return {'state': state, 'memory_update': memory_update}
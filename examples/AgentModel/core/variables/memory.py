from abc import abstractmethod

from ..base import AgentVariable, AgentModule

class BaseMemory(AgentVariable):
    @abstractmethod
    def step(self, *args, **kwargs): ...


class PromptedMemory(BaseMemory, AgentModule):
    def __init__(self, identity, llm, prompt_template):
        super().__init__()
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template
        self.reset()
        
    def reset(self):
        self.memory_history = ['Beginning of task.']
        self.memory_content = self.memory_history[-1]
        
    def step(self):
        self.memory_content = self.memory_history[-1]
        
    def __call__(self, state, intent, llm_kwargs=None, **kwargs):
        if llm_kwargs is None:
            llm_kwargs = {}
            
        user_prompt = self.prompt_template.format(
            memory=self.memory_content, 
            state=state, intent=intent,
            **kwargs
        )
        llm_output = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **llm_kwargs
        )
        return llm_output

    def update(self, state, intent, **kwargs):
        llm_output = self(state, intent, **kwargs)
        self.memory_history.append(llm_output['updated_memory'])
    
    def get_value(self):
        return f"""\
# Memory:
{self.memory_content}\
"""


class StepKeyValueMemory(BaseMemory):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.reset()

    def reset(self):
        self.history = []
        self.current_step = dict()

    def step(self):
        self.history.append(self.current_step)
        self.current_step = dict()

    def update(self, **kwargs):
        self.current_step.update(kwargs)

    def get_value(self):
        memory_lines = ['# History:']
        # memory_lines = []
        if not self.history:
            memory_lines.append('Beginning of interaction.')

        for i, step in enumerate(self.history):
            step_lines = [f'## Step {i + 1}']
            for key in self.keys:
                step_lines.append(f'### {key.capitalize()}:\n{step[key]}')
            memory_lines.append('\n'.join(step_lines))

        return '\n\n'.join(memory_lines)
        # memory = '\n\n'.join(memory_lines)

        # state_lines = ['# Current State:']


#         state_lines = []
#         for key in self.keys:
#             state_lines.append(f'## {key.capitalize()}:\n{self.current_step[key]}')

#         state = '\n'.join(state_lines)

#         return f"""\
# # History:

# {memory}

# # Current State:

# {state}\
# """


class StepPromptedMemory(StepKeyValueMemory):
    def __init__(self, identity, llm, prompt_template, keys, memory_key='memory_update'):
        super().__init__(keys)
        assert memory_key not in keys
        self.identity = identity
        self.llm = llm
        self.prompt_template = prompt_template
        self.memory_key = memory_key
        self.reset()
        
    def update(self, llm_kwargs=None, **kwargs): 
        if llm_kwargs is None:
            llm_kwargs = {}
            
        user_prompt = self.prompt_template.format(
            memory=self.get_value(), 
            **kwargs
        )
        
        llm_output = self.llm(
            system_prompt=str(self.identity), user_prompt=user_prompt, **llm_kwargs
        )
        self.current_step[self.memory_key] = llm_output[self.memory_key]
        
        super().update(**kwargs)
        
    def get_value(self):
        memory_lines = ['# History:']
        # memory_lines = []
        if not self.history:
            memory_lines.append('Beginning of interaction.')

        for i, step in enumerate(self.history):
            step_lines = [f'## Step {i + 1}']
            step_lines.append(f'### State:\n{step[self.memory_key]}')
            for key in self.keys:
                step_lines.append(f'### {key.capitalize()}:\n{step[key]}')
            memory_lines.append('\n'.join(step_lines))

        return '\n\n'.join(memory_lines)
        
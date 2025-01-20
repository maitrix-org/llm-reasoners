from abc import abstractmethod

from ..base import AgentVariable


class BaseMemory(AgentVariable):
    @abstractmethod
    def step(self, *args, **kwargs): ...


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
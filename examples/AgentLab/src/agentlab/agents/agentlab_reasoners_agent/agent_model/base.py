from abc import abstractmethod


class AgentVariable:
    @abstractmethod
    def update(self, *args, **kwargs): ...

    @abstractmethod
    def get_value(self, *args, **kwargs): ...

    @abstractmethod
    def reset(self, *args, **kwargs): ...

    def __str__(self):
        return self.get_value()


class AgentModule:
    @abstractmethod
    def __call__(self, *args, **kwargs): ...


class Agent:
    @abstractmethod
    def step(self, *args, **kwargs): ...

    @abstractmethod
    def reset(self, *args, **kwargs): ...


class Environment:
    @abstractmethod
    def get_obs(self, *args, **kwargs): ...

    @abstractmethod
    def step(self, *args, **kwargs): ...

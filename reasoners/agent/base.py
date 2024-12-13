from abc import abstractmethod

class AgentVariable:
    @abstractmethod
    def update(self, *args, **kwargs): ...

    @abstractmethod
    def get_value(self, *args, **kwargs): ...

    def reset(self, *args, **kwargs):
        pass

    def __str__(self):
        return self.get_value()
    

class AgentModule:
    @abstractmethod
    def __call__(self, *args, **kwargs): ...
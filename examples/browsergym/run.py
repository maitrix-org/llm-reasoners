from reasoners import SearchConfig, WorldModel, LanguageModel, Reasoner
from reasoners.lm import OpenAIModel
from browsergym import BrowserGym

# Initialize components with proper error handling
env = BrowserGym("webarena.313")
llm = OpenAIModel()
agent = Agent(llm)


class AgentReasoner(Reasoner):
    def __init__(
        self,
        env,
        agent,
        algorithm: str,
        world_model: WorldModel,
        search_config: SearchConfig,
    ):
        """
        Initialize Reasoner with required components.

        Args:
            env: BrowserGym environment
            agent: Agent instance
            algorithm: String specifying the algorithm (e.g., "MCTS")
            world_model: WorldModel instance
            search_config: SearchConfigBrowsergym instance
        """
        self.world_model = world_model if world_model else WorldModel(env)
        self.search_config = search_config if search_config else SearchConfig(agent)
        if not isinstance(algorithm, str):
            raise ValueError("Algorithm must be a string")
        self.algorithm = self._get_algorithm(algorithm)

    def __call__(self, task_id: str):
        """
        Execute reasoning process for given task.

        Args:
            task_id: String identifier for the task
        """
        if not isinstance(task_id, str):
            raise ValueError("Task ID must be a string")
        return self.algorithm(self.world_model, self.search_config)


reasoner = AgentReasoner(env, agent, algorithm="MCTS")
reasoner()

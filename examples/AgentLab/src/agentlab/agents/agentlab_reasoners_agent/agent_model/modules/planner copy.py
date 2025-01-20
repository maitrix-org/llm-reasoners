import copy

from ..base import AgentModule
from .planner_utils import SearchConfigWrapper, WorldModelWrapper
from reasoners import Reasoner
from reasoners.algorithm import DFS, MCTS

import logging

logger = logging.getLogger("test")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class BasePlanner(AgentModule): ...


class PolicyPlanner(AgentModule):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def __call__(self, state, memory):
        intent_response = self.policy(state=state, memory=memory)[0]  # returns a list now
        # intent_response = self.policy(state=state, memory=memory)
        return intent_response


class LLMReasonerPlanner(AgentModule):
    def __init__(self, policy, world_model, critic, algorithm, **kwargs):
        super().__init__()
        self.policy = policy
        self.world_model = world_model
        self.critic = critic

        self.reasoner_world_model = WorldModelWrapper(world_model, **kwargs)
        self.reasoner_search_config = SearchConfigWrapper(policy, critic, **kwargs)
        # self.reasoner_search_algo = MCTS(
        #     output_trace_in_each_iter=True, disable_tqdm=False)
        # self.reasoner_search_algo = DFS(max_per_state=5, depth=1, prior=False)
        self.reasoner_search_algo = algorithm
        self.reasoner = Reasoner(
            world_model=self.reasoner_world_model,
            search_config=self.reasoner_search_config,
            search_algo=self.reasoner_search_algo,
        )

        self.logger = None

    def __call__(self, state, memory):
        # We need to define the llm reasoner
        example = {"state": state, "memory": copy.deepcopy(memory)}
        self.reasoner_world_model.logger = self.logger
        self.reasoner_search_config.logger = self.logger

        result = self.reasoner(example)
        intent = result.terminal_state["action_history"][0]
        # adding in result for visualization
        return {"intent": intent, "planner_algorithm_output": result}

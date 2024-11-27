import copy

from ..base import AgentModule
from .planner_utils import SearchConfigWrapper, WorldModelWrapper
from reasoners import Reasoner
from reasoners.algorithm import DFS


class BasePlanner(AgentModule): ...


class PolicyPlanner(AgentModule):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def __call__(self, state, memory, **kwargs):
        intent_response = self.policy(state=state, memory=memory, **kwargs)
        return intent_response


class LLMReasonerPlanner(AgentModule):
    def __init__(self, policy, world_model, critic, 
                 search_num_actions, search_depth, critic_num_samples, 
                 llm_base_url, llm_api_key,
                 **kwargs):
        super().__init__()
        self.policy = policy
        self.world_model = world_model
        self.critic = critic

        self.reasoner_world_model = WorldModelWrapper(world_model)
        self.reasoner_search_config = SearchConfigWrapper(policy, critic, 
                                                          policy_freq_top_k=search_num_actions,
                                                          critic_n=critic_num_samples,
                                                          search_depth=search_depth,
                                                          llm_base_url=llm_base_url,
                                                          llm_api_key=llm_api_key)
        # self.reasoner_search_algo = MCTS(output_trace_in_each_iter=True, disable_tqdm=False)
        self.reasoner_search_algo = DFS(max_per_state=search_num_actions, 
                                        depth=search_depth, 
                                        prior=False)
        self.reasoner = Reasoner(
            world_model=self.reasoner_world_model,
            search_config=self.reasoner_search_config,
            search_algo=self.reasoner_search_algo,
        )

        self.logger = None

    def __call__(self, state, memory, **kwargs):
        # We need to define the llm reasoner
        example = {'state': state, 'memory': copy.deepcopy(memory)}
        example.update(kwargs)
        self.reasoner_world_model.logger = self.logger
        self.reasoner_search_config.logger = self.logger

        result = self.reasoner(example)
        intent = result.terminal_state['action_history'][0]
        return {'intent': intent}

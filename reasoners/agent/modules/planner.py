from abc import abstractmethod
import copy

from ..base import AgentModule
from .policy import Policy
from .world_model import WorldModel
from .critic import Critic
from .planner_utils import SearchConfigWrapper, WorldModelWrapper
from reasoners import Reasoner
from reasoners.algorithm import DFS


class Planner(AgentModule):
    @abstractmethod
    def __call__(self, state, memory, *args, **kwargs): ...


class PolicyPlanner(Planner):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def __call__(self, state, memory, **kwargs):
        plan_response = self.policy(state=state, memory=memory, **kwargs)
        return plan_response


class ReasonerPlanner(Planner):
    def __init__(self, policy: Policy, world_model: WorldModel, critic: Critic, 
                 search_num_actions: int, search_depth: int, 
                 policy_num_samples: int, policy_output_name: str, 
                 critic_num_samples: int, 
                 llm_base_url: str, llm_api_key: str,
                 **kwargs):
        super().__init__()
        self.policy = policy
        self.world_model = world_model
        self.critic = critic
        self.policy_num_samples = policy_num_samples
        self.policy_output_name = policy_output_name

        self.reasoner_world_model = WorldModelWrapper(world_model, action_name=self.policy_output_name)
        self.reasoner_search_config = SearchConfigWrapper(policy, critic, 
                                                          policy_n=policy_num_samples,
                                                          policy_freq_top_k=search_num_actions,
                                                          policy_output_name=policy_output_name,
                                                          critic_n=critic_num_samples,
                                                          search_depth=search_depth,
                                                          llm_base_url=llm_base_url,
                                                          llm_api_key=llm_api_key)
        # self.reasoner_search_algo = MCTS(output_trace_in_each_iter=True, disable_tqdm=False)
        self.reasoner_search_algo = DFS(max_per_state=search_num_actions,
                                        depth=search_depth,
                                        prior=False,
                                        return_if_single_first_action=True,
                                        use_mp=True)
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
        cur_node = result.terminal_nodes[0]
        while cur_node.depth > 1: # go to the first layer
            cur_node = cur_node.parent
        
        plan = cur_node.action['action']
        return {self.policy_output_name: plan}

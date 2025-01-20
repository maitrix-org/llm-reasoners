import copy
import logging
from typing import Callable, Literal
from functools import partial
from dataclasses import dataclass

import numpy as np

# Reasoners imports
from reasoners import Reasoner, WorldModel, SearchConfig
from reasoners.lm.openai_model_w_parser import OpenAIModel
from reasoners.algorithm import MCTS, DFS, BeamSearch
from .agent_model.modules import PromptedPolicy, PromptedWorldModel, PromptedCritic
from .agent_model.variables import AgentInstructionEnvironmentIdentity
from .agent_model.agent_prompts import (
    policy_prompt_template as DefaultPolicyPromptTemplate,
    world_model_prompt_template as DefaultWorldModelPromptTemplate,
    critic_prompt_template as DefaultCriticPromptTemplate,
)

# AgentLab imports
from agentlab.llm.chat_api import BaseModelArgs

# Adaptive AgentLab imports
from .utils import llm_response_parser, cluster_actions


logger = logging.getLogger("planner")


@dataclass
class SearchAlgorithmArgs:
    algorithm: str = "MCTS"
    output_trace_in_each_iter: bool = False
    w_exp: float = 1.0
    depth_limit: int = 5
    n_iters: int = 10
    cum_reward: Callable[[list[float]], float] = sum
    calc_q: Callable[[list[float]], float] = np.mean
    simulate_strategy: str | Callable[[list[float]], int] = "max"
    output_strategy: str = "max_reward"
    uct_with_fast_reward: bool = True


@dataclass
class PolicyArgs:
    mode: str = "default"
    # TODO: add more args beyond default


@dataclass
class WorldModelArgs:
    mode: str = "default"
    # TODO: add more args beyond default


@dataclass
class CriticArgs:
    mode: str = "default"
    # TODO: add more args beyond default


@dataclass
class PlannerLLMArgs:
    model: str = None
    max_tokens: int = 16_384
    temperature: float = 0.7
    additional_prompt: str = None
    backend: Literal["openai", "sglang"] = "openai"
    is_instruct_model: bool = True


@dataclass
class PlannerArgs:
    search_algorithm_args: SearchAlgorithmArgs = None
    policy_args: PolicyArgs = None
    world_model_args: WorldModelArgs = None
    critic_args: CriticArgs = None
    llm_args: PlannerLLMArgs = None


def make_algorithm(args: SearchAlgorithmArgs):
    if args.algorithm == "MCTS":
        return MCTS(
            output_trace_in_each_iter=args.output_trace_in_each_iter,
            w_exp=args.w_exp,
            depth_limit=args.depth_limit,
            n_iters=args.n_iters,
            cum_reward=args.cum_reward,
            calc_q=args.calc_q,
            simulate_strategy=args.simulate_strategy,
            output_strategy=args.output_strategy,
            uct_with_fast_reward=args.uct_with_fast_reward,
        )
    else:
        # TODO: support other algorithms from llm-reasoners
        raise ValueError(f"Algorithm {args.algorithm} not supported")


class Planner:
    """The Planner class.
    Given the agent's observation, it plans a sequence of actions via search, e.g., MCTS."""

    def __init__(self, policy, world_model, critic, algorithm, **kwargs):
        super().__init__()
        self.reasoner = Reasoner(
            dynamics=WorldModelWrapper(world_model, max_plan_steps=kwargs["max_plan_steps"]),
            search_config=SearchConfigWrapper(policy, critic, **kwargs),
            search_algo=algorithm,
        )

    def __call__(self, state, memory):
        example = {
            "state": state,
            "memory": copy.deepcopy(memory),
        }  # Note: `example` is actually context for the planner
        result = self.reasoner(example)
        return {
            "action_plan": result.terminal_state["action_history"],
            "plan_full_result": result,
        }

    @classmethod
    def from_args(cls, args: PlannerArgs, identity: AgentInstructionEnvironmentIdentity = None):
        search_algorithm = make_algorithm(args.search_algorithm_args)
        llm = OpenAIModel(
            model=args.llm_args.model,
            backend=args.llm_args.backend,
            is_instruct_model=args.llm_args.is_instruct_model,
            max_tokens=args.llm_args.max_tokens,
            temperature=args.llm_args.temperature,
            additional_prompt=args.llm_args.additional_prompt,
        )

        if args.policy_args.mode == "default":
            policy = PromptedPolicy(
                identity=identity,
                llm=llm,
                prompt_template=DefaultPolicyPromptTemplate,
                parser=partial(llm_response_parser, keys=["intent"], optional_keys=["think"]),
            )
        else:
            raise ValueError(f"Policy mode {args.policy_args.mode} not supported")

        if args.world_model_args.mode == "default":
            world_model = PromptedWorldModel(
                identity=identity,
                llm=llm,
                prompt_template=DefaultWorldModelPromptTemplate,
                parser=partial(llm_response_parser, keys=["next_state"]),
            )
        else:
            raise ValueError(f"World Model mode {args.world_model_args.mode} not supported")

        if args.critic_args.mode == "default":
            critic = PromptedCritic(
                identity=identity,
                llm=llm,
                prompt_template=DefaultCriticPromptTemplate,
                parser=partial(
                    llm_response_parser,
                    keys=["status", "on_the_right_track"],
                    optional_keys=["think"],
                ),
            )
        else:
            raise ValueError(f"Critic mode {args.critic_args.mode} not supported")

        return cls(
            policy=policy,
            world_model=world_model,
            critic=critic,
            algorithm=search_algorithm,
            max_plan_steps=args.search_algorithm_args.depth_limit,
        )


class WorldModelWrapper(WorldModel):
    """The WorldModelWrapper class. It wraps the WorldModel class in llm-reasoners to override.
    Given a state and action, it predicts the next state.
    """

    def __init__(self, world_model, max_plan_steps=2):
        super().__init__()
        self.world_model = world_model  # Note: This is a prompted llm, not the `WorldModel` class in LLM-Reasoners. This is a bit confusing there is a `PromptedWorldModel` and a `WorldModel`` class in LLM-Reasoners.
        self.max_plan_steps = max_plan_steps

    def init_state(self):
        return {
            "memory": copy.deepcopy(self.example["memory"]),
            "summary_state": self.example["state"],
            "action_history": [],
            "step_idx": 0,
        }

    def step(self, state, action):
        llm_output = self.world_model(state["summary_state"], state["memory"], action["action"])

        next_state = {
            "summary_state": llm_output["next_state"],
            "memory": copy.deepcopy(state["memory"]),
            "action_history": state["action_history"] + [action["action"]],
            "step_idx": state["step_idx"] + 1,
        }

        logger.debug("Planner world model finished. Here is the next_state:")
        logger.debug(next_state)

        next_state["memory"].update(state=state["summary_state"], intent=action["action"])
        next_state["memory"].step()

        return next_state, {"next_state": next_state}

    def is_terminal(self, state):
        # TODO: Add more sophisticated terminal condition, e.g., reward threshold
        return state["step_idx"] > self.max_plan_steps

    def update_example(self, example, **kwargs):
        super().update_example(example, **kwargs)


class SearchConfigWrapper(SearchConfig):
    """The SearchConfigWrapper class. It wraps the SearchConfig class in llm-reasoners to override.
    It contains the policy and critic (reward function here) in `Planner` to get the actions and rewards required in search.
    """

    def __init__(
        self,
        policy,
        critic,
        policy_temperature=1.0,
        policy_top_p=0.95,
        policy_n=20,
        policy_freq_top_k=5,
        critic_temperature=1.0,
        critic_top_p=0.95,
        critic_n=20,
        **kwargs,
    ):
        super().__init__()
        self.policy = policy
        self.critic = critic

        self.policy_temperature = policy_temperature
        self.policy_top_p = policy_top_p
        self.policy_n = policy_n
        self.policy_freq_top_k = policy_freq_top_k

        self.critic_temperature = critic_temperature
        self.critic_top_p = critic_top_p
        self.critic_n = critic_n

    def get_actions(self, state):
        llm_output = self.policy(
            state["summary_state"],
            state["memory"],
            llm_kwargs={
                "temperature": self.policy_temperature,
                "top_p": self.policy_top_p,
                "n": self.policy_n,
            },
        )

        action2freqs = {}
        for ans_dict in llm_output:  # structured to just be ans_dicts
            if ans_dict is None:
                continue
            action = ans_dict.get("intent", None)
            if action is not None:
                freq, _ = action2freqs.get(action, (0, ""))
                action2freqs[action] = (freq + 1, ans_dict.get("think", None))

        logger.debug("Planner policy finished. Here is the action2freqs:")
        logger.debug(action2freqs)

        cluster2freqs = {}
        while len(cluster2freqs) == 0:
            cluster2freqs = cluster_actions(
                llm=self.policy.llm,
                action2freqs=action2freqs,
            )

        action_freq_thoughts = [
            (action, freq, think) for action, (freq, think) in cluster2freqs.items()
        ]
        action_freq_thoughts.sort(key=lambda x: -x[1])
        action_freq_thoughts = action_freq_thoughts[: self.policy_freq_top_k]

        logger.debug("Planner cluster actions finished. Here is the action_freq_thoughts:")
        logger.debug(action_freq_thoughts)

        action_outputs = [
            {"action": action, "freq": freq, "think": think}
            for action, freq, think in action_freq_thoughts
        ]

        return action_outputs

    def fast_reward(self, state, action):
        llm_output = self.critic(
            state["summary_state"],
            state["memory"],
            llm_kwargs={
                "temperature": self.critic_temperature,
                "top_p": self.critic_top_p,
                "n": self.critic_n,
            },
        )

        """Note: Assuming the following response format:
        Thoughts: <your thoughts and reasoning process>
        Status: "task_goal_reached" or "task_goal_not_reached"
        On the right track to success: “yes” or “no”
        """
        scores = []
        thoughts = []
        for ans_dict in llm_output:
            if ans_dict["status"] == "task_goal_reached":
                score = 10
            elif ans_dict["on_the_right_track"] == "yes":
                score = 1
            else:
                score = 0
            scores.append(score)
            thoughts.append(ans_dict.get("think"))
        reward = sum(scores) / len(scores)

        logger.debug("Planner critic finished. Here is the scores and thoughts:")
        logger.debug(scores)
        logger.debug(thoughts)

        return reward, {"scores": scores, "thoughts": thoughts}

    def reward(self, state, action, **kwargs):
        return sum(kwargs["scores"]) / len(kwargs["scores"]), kwargs

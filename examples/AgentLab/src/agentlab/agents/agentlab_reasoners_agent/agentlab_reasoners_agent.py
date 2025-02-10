from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from warnings import warn
from typing import Callable
import numpy as np
from PIL import Image
import io
import base64

# AgentLab & BrowserGym imports
import bgym
from browsergym.experiments.agent import Agent, AgentInfo
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments.benchmark import Benchmark, HighLevelActionSetArgs
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs, make_system_message, make_user_message
from agentlab.llm.llm_utils import (
    Discussion,
    ParseError,
    SystemMessage,
    retry,
    parse_html_tags_raise,
)
from agentlab.llm.tracking import cost_tracker_decorator

from .agentlab_reasoners_agent_prompt import (
    EncoderPrompt,
    ReasonersPromptFlags,
    PolicyPrompt,
    ActorPrompt,
)

from ..generic_agent.generic_agent import GenericAgentArgs, GenericAgent

# Reasoners imports
from reasoners import (
    LanguageModel,
)
from reasoners.algorithm import MCTS, DFS, BeamSearch
from .agent_model.modules import (
    LLMReasonerPlanner,
    PolicyPlanner,
    PromptedActor,
    PromptedCritic,
    PromptedEncoder,
    PromptedPolicy,
    PromptedWorldModel,
)
from .agent_model.variables import (
    AgentInstructionEnvironmentIdentity,
    BrowserActionSpace,
    BrowserGymObservationSpace,
    StepKeyValueMemory,
)
from .agent_model.agent_prompts import (
    encoder_prompt_template,
    policy_prompt_template,
    world_model_prompt_template,
    critic_prompt_template,
)
from typing import Any
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
class PlannerLLMArgs(BaseModelArgs):
    pass


@dataclass
class PlannerArgs:
    search_algorithm_args: SearchAlgorithmArgs = None
    llm_args: PlannerLLMArgs = None


@dataclass
class ReasonersAgentArgs(GenericAgentArgs):
    flags: ReasonersPromptFlags = None
    planner_args: PlannerArgs = None
    action_set_args: HighLevelActionSetArgs = None
    max_retry: int = 1
    # observation_type: VisualWebArenaObservationType = "axtree_som"

    def __post_init__(self):
        logger.info(f"ReasonersAgentArgs.__post_init__: {self.action_set_args}")
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"ReasonersAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return ReasonersAgent(
            chat_model_args=self.chat_model_args,
            planner_args=self.planner_args,
            action_set=self.action_set_args.make_action_set(),
            flags=self.flags,
            max_retry=self.max_retry,
        )


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"


def llm_response_parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ""


class ReasonersAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        planner_args: PlannerArgs,
        action_set: HighLevelActionSet,
        flags: ReasonersPromptFlags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        logger.info("ReasonersAgent.__init__: Initializing ReasonersAgent")
        # <from reaonsers>
        self.action_set = action_set
        # </from reaonsers>
        self.make_misc(args=planner_args)
        self.make_planner(args=planner_args)

    def make_misc(self, args: PlannerArgs):
        logger.info("ReasonersAgent.make_misc: Making miscellaneous components")
        if "gpt-4o" in args.llm_args.model_name:
            # from .agent_model.llms import OpenDevinParserLLM
            # from openai import OpenAI
            # import os

            # self.planner_llm = OpenDevinParserLLM(
            #     opendevin_llm=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            # )
            # self.planner_llm = args.llm_args.make_model()
            from .agent_model.llms import LLM
            import os

            self.planner_llm = LLM(
                model=args.llm_args.model_name,
                api_key=os.environ["OPENAI_API_KEY"],
            )
        else:
            raise ValueError(f"Model {args.llm_args.model_name} not supported")

        # <from mingkai>
        logger.info("ReasonersAgent.make_misc: Making observation space")
        self.observation_space = BrowserGymObservationSpace()  # TODO: @zj should be passed in
        agent_name = "Web Browsing Agent"
        agent_description = "An information and automation assistant who responds to \
user instructions by browsing the internet. The assistant strives to answer each question \
accurately, thoroughly, efficiently, and politely, and to be forthright when it is \
impossible to answer the question or carry out the instruction. The assistant will \
end the task once it sends a message to the user."
        logger.info("ReasonersAgent.make_misc: Making identity")
        self.identity = AgentInstructionEnvironmentIdentity(
            agent_name=agent_name,
            agent_description=agent_description,
            observation_space=self.observation_space,
            action_space=self.action_set.describe(with_long_description=True, with_examples=True),
        )
        logger.info("ReasonersAgent.make_misc: Making encoder")
        self.encoder = PromptedEncoder(
            self.identity,
            self.planner_llm,
            prompt_template=encoder_prompt_template,
            parser=partial(llm_response_parser, keys=["state"]),
        )
        logger.info("ReasonersAgent.make_misc: Making memory")
        if self.flags.use_intent_only_memory:
            self.memory = StepKeyValueMemory(["intent"])
        else:
            self.memory = StepKeyValueMemory(["state", "intent"])
        # </from mingkai>

    def make_planner(
        self,
        args: PlannerArgs,
    ):
        """Construct the planner."""

        # <from reaonsers>
        def _make_algorithm(args: SearchAlgorithmArgs):
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
                raise ValueError(f"Algorithm {args.algorithm} not supported")

        logger.info("ReasonersAgent.make_planner: Making search algorithm")
        search_algorithm = _make_algorithm(args.search_algorithm_args)
        # </from reaonsers>

        # <from mingkai>
        policy = PromptedPolicy(
            self.identity,
            self.planner_llm,
            prompt_template=policy_prompt_template,
            parser=partial(llm_response_parser, keys=["intent"], optional_keys=["think"]),
        )
        world_model = PromptedWorldModel(
            self.identity,
            self.planner_llm,
            prompt_template=world_model_prompt_template,
            parser=partial(llm_response_parser, keys=["next_state"]),
        )
        critic = PromptedCritic(
            self.identity,
            self.planner_llm,
            prompt_template=critic_prompt_template,
            parser=partial(
                llm_response_parser, keys=["status", "on_the_right_track"], optional_keys=["think"]
            ),
        )

        logger.info("ReasonersAgent.make_planner: Making planner")
        self.planner = LLMReasonerPlanner(
            policy=policy, world_model=world_model, critic=critic, algorithm=search_algorithm
        )
        # </from mingkai>

    def obs_preprocessor(self, obs: dict) -> Any:
        # Optionally override this method to customize observation preprocessing
        # The output of this method will be fed to the get_action method and also saved on disk.
        return super().obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs):
        # Processing obs
        self.obs_history.append(obs)
        obs = self.obs_preprocessor(obs)

        # Post-processing for obs for Encoder
        logger.info("ReasonersAgent.get_action: Updating identity")
        self.identity.update(user_instruction=obs["goal_object"][0]["text"])

        # Encode(Summarize) the obs into a textual state, to be used by the planner
        summary_state = self.encoder(
            obs["axtree_txt"], image_to_jpg_base64_url(obs["screenshot_som"]), self.memory
        )["state"]
        logger.info(f"*Summary State*: {summary_state}")

        logger.info("ReasonersAgent.get_action: Running planner")
        print(f"-----Running planner------")
        plan_result_dict = self.planner(summary_state, self.memory)
        # intent = planner_result_dict["intent"]
        plan_steps, plan_full_result = (
            plan_result_dict["plan_steps"],  # a list of planned actions
            plan_result_dict["plan_full_result"],  # planner Result class
        )
        plan_steps_string = "\n".join(
            [f"{i}. {step}" for i, step in enumerate(plan_steps, start=1)]
        )
        # planner_algorithm_output = planner_result_dict["planner_algorithm_output"]
        print(f"==========Plan steps from AgentLab Reasoners Agent==========")
        print(plan_steps)
        print(f"==========Plan full result from AgentLab Reasoners Agent==========")
        print(plan_full_result)
        # return None

        print(f"-----Running actor------")
        max_prompt_tokens, max_trunc_itr = self._get_maxes()
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        actor_prompt = ActorPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
            # adding in information from the encoding step
            # summary_state=summary_state,
            plan_steps_string=plan_steps_string,  # the intent (next action) from the planner
        )
        logger.info(f"*System prompt*: {system_prompt}")
        print(f"==========System prompt from AgentLab Reasoners Agent==========")
        print(system_prompt)
        print(f"==========Actor prompt from AgentLab Reasoners Agent==========")
        print(actor_prompt)

        actor_human_prompt = dp.fit_tokens(
            shrinkable=actor_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        # logger.info(f"*Actor human prompt*: {actor_human_prompt}")
        print(f"==========Actor human prompt from AgentLab Reasoners Agent==========")
        print(actor_human_prompt)
        print("type of system_prompt: ", type(system_prompt))
        print("type of actor_prompt: ", type(actor_prompt))
        print("type of actor_human_prompt: ", type(actor_human_prompt))
        try:
            actor_chat_messages = Discussion([system_prompt, actor_human_prompt])
            # logger.info(f"*Actor chat messages*: {actor_chat_messages}")
            print(f"==========Actor chat messages from AgentLab Reasoners Agent==========")
            print(actor_chat_messages)
            actor_ans_dict = retry(
                self.chat_llm,
                actor_chat_messages,
                n_retry=self.max_retry,
                parser=actor_prompt._parse_answer,
            )
            logger.info(f"*Actor ans dict*: {actor_ans_dict}")
            print(f"==========Actor ans dict from AgentLab Reasoners Agent==========")
            print(actor_ans_dict)
            actor_ans_dict["busted_retry"] = 0
            actor_ans_dict["n_retry"] = (len(actor_chat_messages) - 3) / 2
        except ParseError as e:
            actor_ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        with open("actor_ans_dict.txt", "w") as f:
            f.write(str(actor_ans_dict))

        stats = self.chat_llm.get_stats()
        logger.info(f"*Stats*: {stats}")
        stats["n_retry"] = actor_ans_dict["n_retry"]
        stats["busted_retry"] = actor_ans_dict["busted_retry"]

        self.plan = plan_steps_string
        self.plan_step = actor_ans_dict.get("step", self.plan_step)  # seems rather redundant
        self.actions.append(actor_ans_dict.get("action", "<empty action>"))
        self.memories.append(actor_ans_dict.get("memory", "<empty memory>"))
        self.thoughts.append(actor_ans_dict.get("think", "<empty think>"))
        logger.info(f"*Thoughts*: {self.thoughts}")
        logger.info(f"*Actions*: {self.actions}")
        logger.info(f"*Memories*: {self.memories}")
        logger.info(f"*Plan*: {self.plan}")
        logger.info(f"*Plan step*: {self.plan_step}")

        agent_info = AgentInfo(
            think=actor_ans_dict.get("think", "<empty think>"),
            # need to merge all the calls together into one massive discussion?
            chat_messages=actor_chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return actor_ans_dict["action"], agent_info


def get_action_post_hoc(agent: ReasonersAgent, obs: dict, ans_dict: dict):
    """
    Get the action post-hoc for the agent.

    This function is used to get the action after the agent has already been run.
    Its goal is to recreate the prompt and the output of the agent a posteriori.
    The purpose is to build datasets for training the agents.

    Args:
        agent (GenericAgent): The agent for which the action is being determined.
        obs (dict): The observation dictionary to append to the agent's history.
        ans_dict (dict): The answer dictionary containing the plan, step, memory, think, and action.

    Returns:
        Tuple[str, str]: The complete prompt used for the agent and the reconstructed output based on the answer dictionary.
    """
    system_prompt = dp.SystemPrompt().prompt

    agent.obs_history.append(obs)

    main_prompt = EncoderPrompt(
        action_set=agent.action_set,
        obs_history=agent.obs_history,
        actions=agent.actions,
        memories=agent.memories,
        thoughts=agent.thoughts,
        previous_plan=agent.plan,
        step=agent.plan_step,
        flags=agent.flags,
    )

    max_prompt_tokens, max_trunc_itr = agent._get_maxes()

    fit_function = partial(
        dp.fit_tokens,
        max_prompt_tokens=max_prompt_tokens,
        model_name=agent.chat_model_args.model_name,
        max_iterations=max_trunc_itr,
    )

    instruction_prompt = fit_function(shrinkable=main_prompt)

    if isinstance(instruction_prompt, list):
        # NOTE: this is when we have images
        instruction_prompt = instruction_prompt[0]["text"]

    # TODO: make sure the bid is in the prompt

    output = ""

    # TODO: validate this
    agent.plan = ans_dict.get("plan", agent.plan)
    if agent.plan != "No plan yet":
        output += f"\n<plan>\n{agent.plan}\n</plan>\n"

    # TODO: is plan_step something that the agent's outputs?
    agent.plan_step = ans_dict.get("step", agent.plan_step)

    memory = ans_dict.get("memory", None)
    agent.memories.append(memory)
    if memory is not None:
        output += f"\n<memory>\n{memory}\n</memory>\n"

    thought = ans_dict.get("think", None)
    agent.thoughts.append(thought)
    if thought is not None:
        output += f"\n<think>\n{thought}\n</think>\n"

    action = ans_dict["action"]
    agent.actions.append(action)
    if action is not None:
        output += f"\n<action>\n{action}\n</action>"

    return system_prompt, instruction_prompt, output

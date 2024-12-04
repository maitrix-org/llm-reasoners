from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from warnings import warn

import bgym
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs, make_system_message, make_user_message
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .reasoners_agent_prompt import EncoderPrompt, ReasonersPromptFlags, PolicyPrompt, ActorPrompt

from ..generic_agent.generic_agent import GenericAgentArgs, GenericAgent

# from .agent_model.modules import (
#     LLMReasonerPlanner,
#     PolicyPlanner,
#     PromptedActor,
#     PromptedCritic,
#     PromptedEncoder,
#     PromptedPolicy,
#     PromptedWorldModel,
# )
# from .agent_model.variables import (
#     AgentInstructionEnvironmentIdentity,
#     BrowserActionSpace,
#     BrowserGymObservationSpace,
#     StepKeyValueMemory,
# )


@dataclass
class ReasonersAgentArgs(GenericAgentArgs):
    flags: ReasonersPromptFlags = None

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            self.agent_name = f"ReasonersAgent-{self.chat_model_args.model_name}".replace(
                "/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return ReasonersAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class ReasonersAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: ReasonersPromptFlags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)

        # there's not really a "prompted" system anymore
        # that's taken care of through the dynamic prompt abstraction

        # self.use_world_model_planning = use_world_model_planning
        # if self.use_world_model_planning:

        #     # generate proposals
        #     self.policy = PromptedPolicy(
        #         self.identity, self.llm, prompt_template=policy_prompt_template, parser=partial(
        #             parser, keys=['intent'], optional_keys=['think'])
        #     )

        #     # predict the next state that results from those proposals
        #     self.world_model = PromptedWorldModel(
        #         self.identity,
        #         self.llm,
        #         prompt_template=world_model_prompt_template,
        #         parser=partial(parser, keys=['next_state'])
        #     )

        #     # evaluate the new state
        #     self.critic = PromptedCritic(
        #         self.identity, self.llm, prompt_template=critic_prompt_template,
        #         parser=partial(
        #             parser, keys=['status', 'on_the_right_track'], optional_keys=['think']
        #         )
        #     )

        #     # self.planner = PolicyPlanner(self.policy)
        #     self.planner = LLMReasonerPlanner(
        #         self.policy, self.world_model, self.critic, algorithm=algorithm)

        # else:
        #     self.policy = PromptedPolicy(
        #         self.identity, self.llm, prompt_template=policy_prompt_template, parser=partial(
        #             parser, keys=['intent'], optional_keys=['think'])
        #     )

        #     self.planner = PolicyPlanner(self.policy)

        # self.actor = PromptedActor(
        #     self.identity, self.llm, prompt_template=actor_prompt_template, parser=partial(
        #         parser, keys=['action'])
        # )

        # self.reset()

    @cost_tracker_decorator
    def get_action(self, obs):
        self.obs_history.append(obs)

        # ENCODER
        # encoder_prompt = EncoderPrompt(
        #     action_set=self.action_set,
        #     obs_history=self.obs_history,
        #     actions=self.actions,
        #     memories=self.memories,
        #     thoughts=self.thoughts,
        #     previous_plan=self.plan,
        #     step=self.plan_step,
        #     flags=self.flags,
        # )
        # max_prompt_tokens, max_trunc_itr = self._get_maxes()

        # system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        # encoder_human_prompt = dp.fit_tokens(
        #     shrinkable=encoder_prompt,
        #     max_prompt_tokens=max_prompt_tokens,
        #     model_name=self.chat_model_args.model_name,
        #     max_iterations=max_trunc_itr,
        #     additional_prompts=system_prompt,
        # )
        # try:
        #     encoder_chat_messages = Discussion(
        #         [system_prompt, encoder_human_prompt])
        #     encoder_ans_dict = retry(
        #         self.chat_llm,
        #         encoder_chat_messages,
        #         n_retry=self.max_retry,
        #         parser=encoder_prompt._parse_answer,
        #     )
        #     encoder_ans_dict["busted_retry"] = 0
        #     encoder_ans_dict["n_retry"] = (len(encoder_chat_messages) - 3) / 2
        # except ParseError as e:
        #     encoder_ans_dict = dict(
        #         action=None,
        #         n_retry=self.max_retry + 1,
        #         busted_retry=1,
        #     )

        # state = encoder_ans_dict["state"]

        # state observation - encoder

        # policy prompt - takes in state from encoder and generates prompt

        # POLICY - LLM Reasoners referencing mingkai's code

        # policy_prompt = PolicyPrompt(
        #     action_set=self.action_set,
        #     obs_history=self.obs_history,
        #     actions=self.actions,
        #     memories=self.memories,
        #     thoughts=self.thoughts,
        #     previous_plan=self.plan,
        #     step=self.plan_step,
        #     flags=self.flags,
        #     # adding in information from the encoding step
        #     state=state
        # )

        # passing in processed planning result from llm reasoners

        # MainPrompt =

        system_prompt = dp.SystemPrompt().prompt
        human_prompt = dp.fit_tokens(
            shrinkable=policy_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        # create discussion object form ssytem and human prompts
        chat_messages = Discussion([system_prompt, human_prompt])

        # retry()

        try:
            policy_chat_messages = Discussion(
                [system_prompt, policy_human_prompt])
            policy_ans_dict = retry(
                self.chat_llm,
                policy_chat_messages,
                n_retry=self.max_retry,
                parser=policy_prompt._parse_answer,
            )
            policy_ans_dict["busted_retry"] = 0
            policy_ans_dict["n_retry"] = (len(encoder_chat_messages) - 3) / 2
        except ParseError as e:
            policy_ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        intent = policy_ans_dict["intent"]
        with open("policy_ans_dict.txt", "w") as f:
            f.write(str(policy_ans_dict))

        # ACTOR
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
            state=state,
            intent=intent
        )

        actor_human_prompt = dp.fit_tokens(
            shrinkable=actor_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            actor_chat_messages = Discussion(
                [system_prompt, actor_human_prompt])
            actor_ans_dict = retry(
                self.chat_llm,
                actor_chat_messages,
                n_retry=self.max_retry,
                parser=actor_prompt._parse_answer,
            )
            actor_ans_dict["busted_retry"] = 0
            actor_ans_dict["n_retry"] = (len(encoder_chat_messages) - 3) / 2
        except ParseError as e:
            actor_ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        with open("actor_ans_dict.txt", "w") as f:
            f.write(str(actor_ans_dict))

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = encoder_ans_dict["n_retry"]
        stats["busted_retry"] = encoder_ans_dict["busted_retry"]

        self.plan = policy_ans_dict.get("intent", self.plan)
        # self.plan_step = encoder_ans_dict.get("step", self.plan_step) # seems rather redundant
        self.actions.append(actor_ans_dict["action"])
        # self.memories.append(encoder_ans_dict.get("memory", None))
        # self.thoughts.append(encoder_ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=policy_ans_dict.get("think", None),
            # need to merge all the calls together into one massive discussion?
            chat_messages=encoder_chat_messages,
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

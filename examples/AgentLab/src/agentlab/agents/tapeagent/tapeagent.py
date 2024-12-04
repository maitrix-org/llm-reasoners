import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bgym

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.tracking import cost_tracker_decorator

##############################
#  TODO: replace this hacky imports after releasing tapeagents and tapeagents[examples] to pypi
try:
    from tapeagents.llms import LiteLLM
    from tapeagents.tools.gym_browser import flatten_axtree
except ImportError as e:
    print("Please run install_tapeagents.sh to install tapeagents first.")
    raise e

import sys

sys.path.append(str(Path(__file__).parent.resolve() / "TapeAgents"))
##############################

from examples.workarena.agent import WorkArenaAgent
from examples.workarena.steps import (
    WorkArenaAction,
    ClickAction,
    GoBackAction,
    GoForwardAction,
    GotoPageAction,
    HoverAction,
    InputTextAction,
    PageObservation,
    PressAction,
    SelectOptionAction,
    ScrollAction,
    WorkArenaTape,
    WorkArenaTask,
    StopStep,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TapeAgentArgs(AgentArgs):
    agent_name: str = "WorkarenaTapeAgent"
    chat_model_args: BaseModelArgs = None

    def make_agent(self) -> bgym.Agent:
        llm = LiteLLM(
            model_name=self.chat_model_args.model_name,
            use_cache=False,
            context_size=self.chat_model_args.max_total_tokens,
            parameters={"temperature": self.chat_model_args.temperature},
        )
        return WorkarenaTapeAgent(llm)

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()


class WorkarenaTapeAgent(bgym.Agent):
    tape: WorkArenaTape

    def __init__(self, llm: LiteLLM):
        self.tapeagent = WorkArenaAgent.create(llm)
        self.tape = WorkArenaTape()

    def obs_preprocessor(self, obs: dict) -> dict:
        axtree = obs.pop("axtree_object")
        obs["axtree_txt"] = flatten_axtree(axtree)
        return obs

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str, bgym.AgentInfo]:
        self.update_tape(obs)
        # run agent and collect thoughts and last action
        tape_segment = []
        action = None
        logger.info(f"Run tape with {len(self.tape)} steps")
        for event in self.tapeagent.run(self.tape):
            if not event.step:
                continue
            step = event.step
            tape_segment.append(step)
            logger.info(f"Generated step: {step.llm_view()}")
            if isinstance(step, WorkArenaAction):
                action = self.step_to_action(step)
        self.tape += tape_segment

        logger.info(f"Action string: {action}")
        return (
            action,
            bgym.AgentInfo(
                extra_info={"tape_segment": [step.model_dump() for step in tape_segment]},
                stats={},
            ),
        )

    def update_tape(self, obs: dict):
        """
        Update tape with new observation
        """
        obs_step = PageObservation(text=obs["axtree_txt"], current_page=1, total_pages=1)
        self.tape = self.tape.append(obs_step)
        if len(self.tape) == 1:  # first observation
            logger.info("First observation, adding goal to tape")
            self.tape = self.tape.append(WorkArenaTask(task=obs["goal"]))

    def step_to_action(self, action: WorkArenaAction) -> str | None:
        """
        Convert action step to an action string with function call
        """
        action_str = ""
        if isinstance(action, GotoPageAction):
            action_str = f"goto('{action.url}')"
        elif isinstance(action, ClickAction):
            action_str = (
                f"click('{action.bid}', button='{action.button}', modifiers={action.modifiers})"
            )
        elif isinstance(action, SelectOptionAction):
            action_str = f"select_option('{action.bid}', '{action.option}')"
        elif isinstance(action, HoverAction):
            action_str = f"hover('{action.bid}')"
        elif isinstance(action, InputTextAction):
            text = action.text.replace("'", "\\'")
            action_str = f"fill('{action.bid}', '{text}')"
        elif isinstance(action, PressAction):
            f"press('{action.bid}', '{action.key_comb}')"
        elif isinstance(action, GoBackAction):
            action_str = "go_back()"
        elif isinstance(action, GoForwardAction):
            action_str = "go_forward()"
        elif isinstance(action, StopStep):
            logger.info("Stopping the loop")
            action_str = None
        elif isinstance(action, ScrollAction):
            action_str = "noop()"  # TODO: implement scroll action
        else:
            raise ValueError(f"Unknown action type: {action}")
        return action_str

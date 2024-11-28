"""Reproducibility Agent


This module contains the classes and functions to reproduce the results of a
study. It is used to create a new study that will run the same experiments as
the original study, but with a reproducibility agent that will mimic the same
answers as the original agent. 

Stats are collected to compare the original agent's answers with the new agent's
answers. Load the this reproducibility study in agent-xray to compare the results.
"""

import difflib
import logging
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import bgym
from browsergym.experiments.agent import AgentInfo
from browsergym.experiments.loop import ExpArgs, ExpResult, yield_all_exp_results
from bs4 import BeautifulSoup
from langchain.schema import AIMessage, BaseMessage
from langchain_community.adapters.openai import convert_message_to_dict

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.dynamic_prompting import ActionFlags
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import make_assistant_message
from agentlab.llm.llm_utils import Discussion, messages_to_dict

from .generic_agent import GenericAgent, GenericAgentArgs


class ReproChatModel:
    """A chat model that reproduces a conversation.

    Args:
        messages (list): A list of messages previously executed.
        delay (int): A delay to simulate the time it takes to generate a response.
    """

    def __init__(self, old_messages, delay=1) -> None:
        self.old_messages = old_messages
        self.delay = delay

    def __call__(self, messages: list | Discussion):
        self.new_messages = copy(messages)

        if len(messages) >= len(self.old_messages):
            # if for some reason the llm response was not saved
            return make_assistant_message("""<action>None</action>""")

        old_response = self.old_messages[len(messages)]
        self.new_messages.append(old_response)
        time.sleep(self.delay)
        # return the next message in the list
        return old_response

    def get_stats(self):
        return {}


@dataclass
class ReproAgentArgs(GenericAgentArgs):

    # starting with "_" will prevent from being part of the index in the load_results function
    _repro_dir: str = None

    def __post_init__(self):
        try:  # some attributes might be temporarily args.CrossProd for hyperparameter generation
            super().__post_init__()
            self.agent_name = f"Repro_{self.agent_name}"
        except AttributeError:
            pass

    def make_agent(self):
        return ReproAgent(self.chat_model_args, self.flags, self.max_retry, self._repro_dir)


class ReproAgent(GenericAgent):

    def __init__(
        self,
        chat_model_args,
        flags,
        max_retry=4,
        repro_dir=None,
    ):
        self.exp_result = ExpResult(repro_dir)
        super().__init__(chat_model_args, flags, max_retry)

    def get_action(self, obs):

        # replace the chat model with a reproducible chat that will mimic the
        # same answers
        step = len(self.actions)
        step_info = self.exp_result.get_step_info(step)
        old_chat_messages = step_info.agent_info.get("chat_messages", None)  # type: Discussion

        if old_chat_messages is None:
            err_msg = self.exp_result.summary_info["err_msg"]

            agent_info = AgentInfo(
                markdown_page=f"Agent had no chat messages. Perhaps there was an error. err_msg:\n{err_msg}",
            )
            return None, agent_info

        # an old bug prevented the response from being saved.
        if len(old_chat_messages) == 2:
            recorded_action = step_info.action
            if recorded_action:
                # Recreate the 3rd message based on the recorded action
                assistant_message = make_assistant_message(f"<action>{recorded_action}</action>")
                old_chat_messages.append(assistant_message)

        self.chat_llm = ReproChatModel(old_chat_messages)
        action, agent_info = super().get_action(obs)

        return _make_agent_stats(
            action, agent_info, step_info, old_chat_messages, self.chat_llm.new_messages
        )


def _make_agent_stats(action, agent_info, step_info, old_chat_messages, new_chat_messages):
    if isinstance(agent_info, dict):
        agent_info = AgentInfo(**agent_info)

    old_msg_str = _format_messages(old_chat_messages)
    new_msg_str = _format_messages(new_chat_messages)

    agent_info.html_page = _make_diff(old_str=old_msg_str, new_str=new_msg_str)
    agent_info.stats.update(_diff_stats(old_msg_str, new_msg_str))

    return action, agent_info


def _format_messages(messages: list[dict]):
    if isinstance(messages, Discussion):
        return messages.to_string()
    messages = messages_to_dict(messages)
    return "\n".join(f"{m['role']} message:\n{m['content']}\n" for m in messages)


def _make_backward_compatible(agent_args: GenericAgentArgs):
    action_set = agent_args.flags.action.action_set
    if isinstance(action_set, (str, list)):
        if isinstance(action_set, str):
            action_set = action_set.split("+")

        agent_args.flags.action.action_set = bgym.HighLevelActionSetArgs(
            subsets=action_set,
            multiaction=agent_args.flags.action.multi_actions,
        )

    return agent_args


def reproduce_study(original_study_dir: Path | str, log_level=logging.INFO):
    """Reproduce a study by running the same experiments with the same agent."""

    original_study_dir = Path(original_study_dir)

    exp_args_list: list[ExpArgs] = []
    for exp_result in yield_all_exp_results(original_study_dir, progress_fn=None):
        agent_args = _make_backward_compatible(exp_result.exp_args.agent_args)
        agent_args = make_repro_agent(agent_args, exp_dir=exp_result.exp_dir)
        exp_args_list.append(
            ExpArgs(
                agent_args=agent_args,
                env_args=exp_result.exp_args.env_args,
                logging_level=log_level,
            )
        )

    # infer benchmark name from task list for backward compatible
    benchmark_name = exp_args_list[0].env_args.task_name.split(".")[0]

    study = Study(
        benchmark=benchmark_name,
        agent_args=[agent_args],
    )
    # this exp_args_list has a different agent_args for each experiment as repro_agent takes the exp_dir as argument
    # so we overwrite exp_args_list with the one we created above
    study.exp_args_list = exp_args_list
    return study


def make_repro_agent(agent_args: AgentArgs, exp_dir: Path | str):
    """Create a reproducibility agent from an existing agent.

    Note, if a new flag was added, it was not saved in the original pickle. When
    loading the pickle it silently adds the missing flag and set it to its
    default value. The new repro agent_args will thus have the new flag set to
    its default value.

    Args:
        agent_args (AgentArgs): The original agent args.
        exp_dir (Path | str): The directory where the experiment was saved.

    Returns:
        ReproAgentArgs: The new agent args.
    """
    exp_dir = Path(exp_dir)
    assert isinstance(agent_args, GenericAgentArgs)
    assert exp_dir.exists()  # sanity check

    return ReproAgentArgs(
        agent_name=f"Repro_{agent_args.agent_name}",
        chat_model_args=agent_args.chat_model_args,
        flags=agent_args.flags,
        max_retry=agent_args.max_retry,
        _repro_dir=exp_dir,
    )


def _make_diff(old_str, new_str):
    page = difflib.HtmlDiff().make_file(
        old_str.splitlines(), new_str.splitlines(), fromdesc="Old Version", todesc="New Version"
    )
    page = page.replace('nowrap="nowrap"', "")  # Remove nowrap attribute
    page = _set_style(page, DIFF_STYLE)
    return page


def _diff_stats(str1: str, str2: str):
    """Try some kind of metrics to make stats about the amount of diffs between two strings."""
    lines1 = str1.splitlines()
    lines2 = str2.splitlines()

    diff = list(difflib.Differ().compare(lines1, lines2))

    # Count added and removed lines
    added = sum(1 for line in diff if line.startswith("+ "))
    removed = sum(1 for line in diff if line.startswith("- "))

    # Calculate difference ratio
    difference_ratio = (added + removed) / (2 * max(len(lines1), len(lines2)))

    return dict(lines_added=added, lines_removed=removed, difference_ratio=difference_ratio)


def _set_style(html_str: str, style: str, prepend_previous_style: bool = False):
    """Add a style tag to an HTML string."""

    soup = BeautifulSoup(html_str, "html.parser")
    style_tag = soup.find("style")

    if not style_tag:
        style_tag = soup.new_tag("style")
        soup.head.append(style_tag)

    current_style = style_tag.string or ""

    if prepend_previous_style:
        style = f"{style}\n{current_style}"
    else:
        style = f"{current_style}\n{style}"

    style_tag.string = style

    return str(soup)


# this is the style to adjust the diff table inside gradio
DIFF_STYLE = """
    table.diff {
        font-size: 10px;
        font-family: Courier;
        border: medium;
        width: 100%;
        max-width: 100%; /* Ensure table does not exceed its container */
        table-layout: auto; /* Adjust column sizes dynamically */
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    /* Constrain the max-width of the 3rd and 6th columns */
    td:nth-child(3), td:nth-child(6) {
        max-width: 200px; /* Adjust this value to suit your content */
        white-space: normal; /* Allow wrapping in content columns */
        overflow-wrap: break-word; /* Break long words/content */
    }
    /* Ensure span elements wrap inside the table */
    .diff_add, .diff_chg, .diff_sub {
        word-wrap: break-word; /* Wrap long text */
        overflow-wrap: break-word;
    }

    /* Keep the rest of the table flexible */
    td {
        white-space: normal; /* Allow wrapping for content */
    }
    .diff_header {
        background-color: #e0e0e0;
    }
    td.diff_header {
        text-align: right;
    }
    .diff_next {
        background-color: #c0c0c0;
    }
    .diff_add {
        background-color: #aaffaa;
    }
    .diff_chg {
        background-color: #ffff77;
    }
    .diff_sub {
        background-color: #ffaaaa;
    }
"""

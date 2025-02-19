"""
Referencing prompt building from. 
https://github.com/ServiceNow/BrowserGym/blob/main/demo_agent/agent.py
"""

import logging
import os
import re
import json
import argparse
import base64
import io
import numpy as np
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.chat import Chat
from browsergym.core.env import BrowserEnv
from browsergym.experiments import EnvArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.core.action.parsers import highlevel_action_parser

from reasoners import SearchConfig, WorldModel, LanguageModel
from .misc import image_to_jpg_base64_url


def get_user_messages_for_current_state(
    obs: dict,
    action_set: HighLevelActionSet, action_history: list[str],
    use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False
) -> list[dict]:
    assert obs["goal_object"], "The goal is missing."
    user_msgs = []
    # goal_object is directly presented as a list of openai-style messages
    user_msgs.extend(obs["goal_object"])
    
    # append action space description
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Action Space

{action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

""",
        }
    )
    
    # append url of all open tabs
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Currently open tabs
    """,
        }
    )
    for page_index, (page_url, page_title) in enumerate(
        zip(obs["open_pages_urls"], obs["open_pages_titles"])
    ):
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
Title: {page_title}
URL: {page_url}
    """,
            }
        )

    # append page AXTree (if asked)
    if use_axtree:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Current page Accessibility Tree

{obs["axtree_txt"]}
""",
            }
        )
    # append page HTML (if asked)
    if use_html:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Current page DOM

{obs["pruned_html"]}

""",
            }
        )

    # append page screenshot (if asked)
    if use_screenshot:
        user_msgs.append(
            {
                "type": "text",
                "text": """\
# Current page Screenshot
    """,
            }
        )
        user_msgs.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(obs["screenshot"]),
                    "detail": "auto",
                },  # Literal["low", "high", "auto"] = "auto"
            }
        )


    # print({action_set.describe(with_long_description=False, with_examples=True)})

    # append past actions (and last error message) if any
    if action_history:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# History of past actions
""",
            }
        )
        user_msgs.extend(
            [
                {
                    "type": "text",
                    "text": f"""\

{action}
""",
                }
                for action in action_history
            ]
        )

        if obs["last_action_error"]:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Error message from last action

{obs["last_action_error"]}

""",
                }
            )

    return user_msgs


def build_propose_prompt(
    obs: dict,
    action_set: HighLevelActionSet, action_history: list[str],
    use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False,
    # logger: logging.Logger = None
) -> tuple[list[dict], list[dict], str]:
    system_msgs = []
    user_msgs = []

    assert obs["goal_object"], "The goal is missing."
    system_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
        }
    )

    user_msgs.extend(get_user_messages_for_current_state(
        obs, action_set, action_history, use_axtree, use_html, use_screenshot))

    # ask for the next action
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action. Make sure to fill in ALL PARAMETERS of the action. 
""",
        }
    )

    prompt_text_strings = []
    for message in system_msgs + user_msgs:
        match message["type"]:
            case "text":
                prompt_text_strings.append(message["text"])
            case "image_url":
                image_url = message["image_url"]
                if isinstance(message["image_url"], dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    prompt_text_strings.append(
                        "image_url: " + image_url[:30] + "... (truncated)"
                    )
                else:
                    prompt_text_strings.append("image_url: " + image_url)
            case _:
                raise ValueError(
                    f"Unknown message type {repr(message['type'])} in the task goal."
                )
    full_prompt_txt = "\n".join(prompt_text_strings)

    return system_msgs, user_msgs, full_prompt_txt


def build_evaluation_prompt(
    obs: dict,
    action: str, action_set: HighLevelActionSet, action_history: list[str],
    use_axtree: bool = True, use_html: bool = False, use_screenshot: bool = False,
    # logger: logging.Logger = None
) -> tuple[list[dict], list[dict], str]:
    system_msgs = []
    user_msgs = []

    assert obs["goal_object"], "The goal is missing."
    system_msgs.append(
        {
            "type": "text",
            "text": """\
# Instructions


Review the current state of the page along with a proposed action and determine how promising it is towards completing the goal. Provide a score between 0 and 100 along with your reasoning in a json object like so:
{
    "reasoning": [your_reasoning]
    "score": [your_score]
}
""",
        }
    )

    user_msgs.extend(get_user_messages_for_current_state(
        obs, action_set, action_history, use_axtree, use_html, use_screenshot))

    # proposed action
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Proposed action

{action}
""",
        }
    )

    # ask for the evaluation
    user_msgs.append(
        {
            "type": "text",
            "text": """\
# Evaluation of Proposed Action

As mentioned before, considering all the information above in the context of the goal, evaluate the proposed action by providing a score from 0 to 10 along with your reasoning. Use a json object like so:
{
    "reasoning": [your_reasoning]
    "score": [your_score]
}
""",
        }
    )

    prompt_text_strings = []
    for message in system_msgs + user_msgs:
        match message["type"]:
            case "text":
                prompt_text_strings.append(message["text"])
            case "image_url":
                image_url = message["image_url"]
                if isinstance(message["image_url"], dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    prompt_text_strings.append(
                        "image_url: " + image_url[:30] + "... (truncated)"
                    )
                else:
                    prompt_text_strings.append("image_url: " + image_url)
            case _:
                raise ValueError(
                    f"Unknown message type {repr(message['type'])} in the task goal."
                )

    full_prompt_txt = "\n".join(prompt_text_strings)

    return system_msgs, user_msgs, full_prompt_txt

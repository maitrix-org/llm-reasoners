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

from reasoners import SearchConfig, WorldModel, LanguageModel


# GENERATING ACTION PROPOSALS


def get_action_proposals(
    obs: dict,
    action_set: HighLevelActionSet,
    action_history: list[str],
    llm: LanguageModel,
    n=10,
    temperature=1.0,
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
    logger: logging.Logger = None,
) -> tuple[str, dict]:

    system_msgs, user_msgs, full_prompt_text = build_propose_prompt(
        obs, action_set, action_history, use_axtree, use_html, use_screenshot
    )

    response = llm.generate(
        full_prompt_text, num_return_sequences=n, temperature=temperature
    )
    action_proposals = response.text

    if logger:
        log_action_proposals(logger, action_proposals)

    return action_proposals, {}


def cluster_actions_proposals(
    actions: list[str], action_set: HighLevelActionSet, logger: logging.Logger = None
) -> list[str]:
    clustered_actions = []
    action_codes = set()
    for action in actions:
        action_code = action_set.to_python_code(action)
        if action_code not in action_codes:
            action_codes.add(action_code)
            clustered_actions.append(action)

    if logger:
        log_clustered_action_proposals(logger, clustered_actions)

    return clustered_actions


def get_clustered_action_proposals(
    obs: dict,
    action_set: HighLevelActionSet,
    action_history: list[str],
    llm: LanguageModel,
    n=10,
    temperature=1.0,
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
    logger: logging.Logger = None,
) -> list[str]:
    actions, info = get_action_proposals(
        obs,
        action_set,
        action_history,
        llm,
        n,
        temperature,
        use_axtree,
        use_html,
        use_screenshot,
        logger=logger,
    )
    clustered_actions = cluster_actions_proposals(actions, action_set, logger=logger)

    return clustered_actions


# EVALUATING ACTIONS


def get_evaluation_of_action_proposal(
    obs: dict,
    action_proposal: str,
    action_set: HighLevelActionSet,
    action_history: list[str],
    llm: LanguageModel,
    n=1,
    temperature=0.25,
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
) -> tuple[str, dict]:

    system_msgs, user_msgs, full_prompt_txt = build_evaluation_prompt(
        obs,
        action_proposal,
        action_set,
        action_history,
        use_axtree,
        use_html,
        use_screenshot,
    )

    response = llm.generate(
        full_prompt_txt, num_return_sequences=n, temperature=temperature
    )
    evaluation = response.text[0]

    return evaluation, {}


def parse_evaluation_json(evaluation: str) -> float:
    json_string = re.search(r"\{.*\}", evaluation, re.DOTALL).group()
    json_object = json.loads(json_string)
    return json_object["score"]


def get_parsed_evaluation_of_action_proposal(
    obs: dict,
    action_proposal: str,
    action_set: HighLevelActionSet,
    action_history: list[str],
    llm: LanguageModel,
    n=1,
    temperature=0.25,
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
    logger: logging.Logger = None,
) -> tuple[float, dict]:
    evaluation, info = get_evaluation_of_action_proposal(
        obs,
        action_proposal,
        action_set,
        action_history,
        llm,
        n,
        temperature,
        use_axtree,
        use_html,
        use_screenshot,
    )
    parsed_evaluation = parse_evaluation_json(evaluation)
    log_evaluation(logger, action_proposal, evaluation, parsed_evaluation)

    return parsed_evaluation, info


def get_parsed_evaluations_of_action_proposals(
    obs: dict,
    action_proposals: list[str],
    action_set: HighLevelActionSet,
    action_history: list[str],
    llm: LanguageModel,
    logger: logging.Logger = None,
) -> list[tuple[str, float]]:
    actions_with_eval = []
    for action_proposal in action_proposals:
        evaluation, info = get_parsed_evaluation_of_action_proposal(
            obs, action_proposal, action_set, action_history, llm, logger=logger
        )
        actions_with_eval.append((action_proposal, evaluation))

    return actions_with_eval


# MANAGING CHAT


def _send_chat_info(chat: Chat, action: str, agent_info: dict):
    """Send the think and action info to the chat."""
    msg = ""
    if "think" in agent_info:
        msg += f"""\
{agent_info["think"]}

"""

    msg += f"""\
action:
{action}
"""

    chat.add_message(role="info", msg=msg)


# HIGH LEVEL ACTION SET DEF


def get_browser_action_set():
    return HighLevelActionSet(
        subsets=["chat", "tab", "nav", "bid", "infeas"],
        strict=False,  # less strict on the parsing of the actions
        multiaction=True,
        demo_mode="off",  # add visual effects # demo_mode doesn't work with webarena. causes an infinite hang.
    )


# MANAGING BROWSERENV


def get_env(task_name, action_set: HighLevelActionSet, seed):
    env_args = EnvArgs(
        task_name=task_name,
        task_seed=seed,
        max_steps=100,
        headless=True,
        record_video=True,
        # viewport={"width": 500, "height": 500},  # can be played with if needed
    )

    env = env_args.make_env(
        action_mapping=action_set.to_python_code,
        exp_dir="./results",
    )
    return env


def reset_env(env: BrowserEnv, seed: int, logger: logging.Logger = None):
    obs, env_info = env.reset(seed=seed)
    obs = obs_preprocessor(obs)
    log_obs(logger, obs, "INITIAL STATE:")
    return obs, env_info


def step_env(env: BrowserEnv, action: str, logger: logging.Logger = None):
    obs, reward, terminated, truncated, step_info = env.step(action)
    obs = obs_preprocessor(obs)
    log_chosen_action(logger, action)
    log_reward(logger, reward)
    log_obs(logger, obs, "NEW STATE:")
    _send_chat_info(env.unwrapped.chat, action, step_info)
    return obs, reward, terminated, truncated, step_info


# when you expand a node, you need to take the node's current action history, make sure that env is aligned with the current state, then you can expand.
def reset_and_replay_actions(env: BrowserEnv, action_history: list[str]) -> BrowserEnv:
    obs, env_info = env.reset(seed=16)
    for action in action_history:
        obs, reward, terminated, truncated, step_info = env.step(action)
        _send_chat_info(env.unwrapped.chat, action, step_info)
    return env


# LOGGING UTILS


def create_logger(task_name: str, out_dir: str = "./logs"):
    logger = logging.getLogger(task_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    handler = logging.FileHandler(f"{out_dir}/{task_name}.html")
    formatter = logging.Formatter("<p>%(asctime)s - %(levelname)s</p>\n%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def log_obs(logger: logging.Logger, obs: dict, text="Current State"):
    logger.info(f"<h3>{text}</h3>\n<img src='{obs['screenshot']}'/>")


def log_reward(logger: logging.Logger, reward: float):
    logger.info(f"<h3>Reward: {reward}</h3>")


def log_action_proposals(logger: logging.Logger, action_proposals: list[str]):
    out = "<h3>Action Proposals</h3>"
    out += "<ul>"
    for action in action_proposals:
        out += f"<li>{action}</li>"
    out += "</ul>"
    logger.info(out)


def log_clustered_action_proposals(
    logger: logging.Logger, clustered_action_proposals: list[str]
):
    out = "<h3>Clustered Action Proposals</h3>"
    out += "<ul>"
    for action in clustered_action_proposals:
        out += f"<li>{action}</li>"
    out += "</ul>"
    logger.info(out)


def log_evaluation(
    logger: logging.Logger, action: str, evaluation: str, parsed_evaluation: float
):
    out = f"<h3>Evaluation of Action:</h3>"
    out += f"<ul>"
    out += f"<li><b>Action</b>: {action}</li>"
    out += f"<li><b>Evaluation</b>: {evaluation}</li>"
    out += f"<li><b>Parsed Evaluation</b>: {parsed_evaluation}</li>"
    out += "</ul>"
    logger.info(out)


def log_chosen_action(logger: logging.Logger, action: str):
    out = f"<h3>Chosen Action</h3>"
    out += f"<ul><li>{action}</li></ul>"
    logger.info(out)


# PROMPT BUILDING UTILS


def get_user_messages_for_current_state(
    obs: dict,
    action_set: HighLevelActionSet,
    action_history: list[str],
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
) -> list[dict]:
    assert obs["goal_object"], "The goal is missing."
    user_msgs = []
    # goal_object is directly presented as a list of openai-style messages
    user_msgs.extend(obs["goal_object"])
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
    action_set: HighLevelActionSet,
    action_history: list[str],
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
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

    user_msgs.extend(
        get_user_messages_for_current_state(
            obs, action_set, action_history, use_axtree, use_html, use_screenshot
        )
    )

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
    action: str,
    action_set: HighLevelActionSet,
    action_history: list[str],
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
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

Review the current state of the page along with a proposed action and determine how promising it is towards completing the goal. Provide a score between 0 and 10 along with your reasoning in a json object like so:
{
    "reasoning": [your_reasoning]
    "score": [your_score]
}
""",
        }
    )

    user_msgs.extend(
        get_user_messages_for_current_state(
            obs, action_set, action_history, use_axtree, use_html, use_screenshot
        )
    )

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


# OTHER MSIC. UTILS


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )

    return parser.parse_args()


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


def obs_preprocessor(obs: dict) -> dict:

    return {
        "chat_messages": obs["chat_messages"],
        # need to convert to base64 to work with llm-reasoners visualizer client
        # don't want a massive list
        "screenshot": image_to_jpg_base64_url(obs["screenshot"]),
        "goal_object": obs["goal_object"],
        "last_action": obs["last_action"],
        "last_action_error": obs["last_action_error"],
        "open_pages_urls": obs["open_pages_urls"],
        "open_pages_titles": obs["open_pages_titles"],
        "active_page_index": obs["active_page_index"],
        "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
    }

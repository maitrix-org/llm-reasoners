import base64
import io
import numpy as np
from PIL import Image

from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.core.action.parsers import highlevel_action_parser


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


# currently hardcoded for the webarena action set
valid_action_types = ["noop", "scroll", "keyboard_press", "click", "fill", "hover", "tab_focus", "new_tab",
                      "go_back", "go_forward", "goto", "tab_close", "select_option", "send_msg_to_user", "report_infeasible"]


def check_validity_of_action_proposal(action_proposal: str):
    """Checks to see if all actions in the proposal exist in the action set. """

    function_calls = highlevel_action_parser.search_string(action_proposal)
    function_calls = sum(function_calls.as_list(), [])

    if len(function_calls) == 0:
        return False

    for function_name, function_args in function_calls:
        if function_name not in valid_action_types:
            return False

    return True

import numpy as np
from PIL import Image
import io
import base64
import json

from reasoners.lm.openai_model_w_parser import OpenAIModel
from agentlab.llm.llm_utils import (
    ParseError,
    parse_html_tags_raise,
)


def llm_response_parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ""


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


def cluster_actions(llm, action2freqs):
    action_candidate_dict = {i: action for i, action in enumerate(action2freqs.keys())}
    action_candidate_json = json.dumps(action_candidate_dict, indent=2)

    input_prompt = _get_cluster_input_template().format(action_candidate_json=action_candidate_json)
    llm_prompt = (
        _get_cluster_instruction_prompt()
        + "\n\n"
        + _get_cluster_example_prompt()
        + "\n\n"
        + input_prompt
    )

    # Run LLM for clustering
    system_prompt = "You are an expert at clustering text."
    generation_output = llm.generate(
        system_prompt=system_prompt,
        prompt=llm_prompt,
        response_format={"type": "json_object"},
    )
    clusters_dict = json.loads(generation_output.text[0])

    cluster2freqs = {}
    for _, cluster_info in clusters_dict.items():
        action = cluster_info.get("intent", None)
        if action is not None:
            cluster2freqs[action] = (0, "")
            for candidate_id in cluster_info.get("candidates", []):
                candidate = action_candidate_dict.get(int(candidate_id), None)
                if candidate is not None:
                    candidate_freq, candidate_think = action2freqs.get(candidate, (0, ""))
            cluster_freq, _ = cluster2freqs[action]
            cluster2freqs[action] = (cluster_freq + candidate_freq, candidate_think)

    return cluster2freqs


def _get_cluster_instruction_prompt():
    return """\
Here is the action space for a browser agent to navigate in a webpage:

16 different types of actions are available:

noop(wait_ms: float = 1000)

send_msg_to_user(text: str)

scroll(delta_x: float, delta_y: float)

fill(bid: str, value: str)

select_option(bid: str, options: str | list[str])

click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])

dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'Meta', 'Shift']] = [])

hover(bid: str)

press(bid: str, key_comb: str)

focus(bid: str)

clear(bid: str)

drag_and_drop(from_bid: str, to_bid: str)

upload_file(bid: str, file: str | list[str])

go_back()

go_forward()

goto(url: str)

Only a single action can be provided at once. Example:
fill('a12', 'example with "quotes"')

Below, you will find lists of intents, or natural language descriptions of actions that, when executed, will translate to one of the function calls above. \
The intents will be provided in the following JSON format:

```json
{
"intent_id": "intent description"
}
```

Your task is to cluster list of intents into semantically equivalent groups, where each group represents intents that lead to the same action when executed \
(i.e., navigating to the Google homepage is translated to goto('https://www.google.com')) and would therefore correspond to the same API call \
in a Playwright browser. Intents that use different wording but convey the same action should be grouped together. Try to minimize the number of clusters.

Represent the clustering results using a JSON object where each cluster has a unique identifier, and each identifier maps to a list of actions in that cluster. \
See below for an abstract example:

```json
{
"cluster_id": {
"intent": "representative intent name for this cluster",
"candidates": [
    "<list of intent ids that belong to this cluster>
]
}
}
```\
"""


def _get_cluster_example_prompt():
    return """\
Concrete Example 1:

Dictionary of Intents:

```json
{
"0": "Navigate to the Google homepage by entering its URL.",
"1": "Go to the Google homepage.",
"2": "Go to the Google homepage",
"3": "Go to the Google homepage by navigating to 'https://www.google.com'",
"4": "Go to the home page of Google"
}
```

["Navigate to the Google homepage by entering its URL.", "Go to the Google homepage.", "Go to the Google homepage", "Go to the Google homepage by navigating to \"https://www.google.com\"", "Go to the home page of Google"]

Clustering Results:

```json
{
"cluster_1": {
"intent": "Navigate to the Google homepage",
"candidates": [0, 1, 2, 3, 4]
}
}
```\
"""


def _get_cluster_input_template():
    return """\
Concrete Example 2:

Dictionary of Intents:

{action_candidate_json}

Clustering Results:
"""

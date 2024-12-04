from functools import cache
import os
import threading
from contextlib import contextmanager

import requests
from langchain_community.callbacks.openai_info import MODEL_COST_PER_1K_TOKENS

TRACKER = threading.local()


class LLMTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def __call__(self, input_tokens: int, output_tokens: int, cost: float):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += cost

    @property
    def stats(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
        }

    def add_tracker(self, tracker: "LLMTracker"):
        self(tracker.input_tokens, tracker.output_tokens, tracker.cost)

    def __repr__(self):
        return f"LLMTracker(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, cost={self.cost})"


@contextmanager
def set_tracker():
    global TRACKER
    if not hasattr(TRACKER, "instance"):
        TRACKER.instance = None
    previous_tracker = TRACKER.instance  # type: LLMTracker
    TRACKER.instance = LLMTracker()
    try:
        yield TRACKER.instance
    finally:
        # If there was a previous tracker, add the current one to it
        if isinstance(previous_tracker, LLMTracker):
            previous_tracker.add_tracker(TRACKER.instance)
        # Restore the previous tracker
        TRACKER.instance = previous_tracker


def cost_tracker_decorator(get_action):
    def wrapper(self, obs):
        with set_tracker() as tracker:
            action, agent_info = get_action(self, obs)
        agent_info.get("stats").update(tracker.stats)
        return action, agent_info

    return wrapper


@cache
def get_pricing_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key, "OpenRouter API key is required"
    # query api to get model metadata
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise ValueError("Failed to get model metadata")

    model_metadata = response.json()
    return {
        model["id"]: {k: float(v) for k, v in model["pricing"].items()}
        for model in model_metadata["data"]
    }


def get_pricing_openai():
    cost_dict = MODEL_COST_PER_1K_TOKENS
    cost_dict = {k: v / 1000 for k, v in cost_dict.items()}
    res = {}
    for k in cost_dict:
        if k.endswith("-completion"):
            continue
        prompt_key = k
        completion_key = k + "-completion"
        if completion_key in cost_dict:
            res[k] = {
                "prompt": cost_dict[prompt_key],
                "completion": cost_dict[completion_key],
            }
    return res

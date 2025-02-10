from reasoners import WorldModel as ReasonersWorldModel
from reasoners import SearchConfig as ReasonersSearchConfig
import copy
import json

from openai import OpenAI

import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

verbose_prompt = True


class WorldModelWrapper(ReasonersWorldModel):
    def __init__(self, world_model, max_steps=2):
        super().__init__()
        self.world_model = world_model
        self.logger = None
        self.max_steps = max_steps

    def init_state(self):
        return {
            "memory": copy.deepcopy(self.example["memory"]),
            "summary_state": self.example["state"],
            "action_history": [],
            "step_idx": 0,
        }

    def step(self, state, action):
        """World Model"""

        llm_output = self.world_model(
            state["summary_state"], state["memory"], action["action"], verbose=verbose_prompt
        )

        next_state = {
            "summary_state": llm_output["next_state"],
            "memory": copy.deepcopy(state["memory"]),
            "action_history": state["action_history"] + [action["action"]],
            "step_idx": state["step_idx"] + 1,
        }

        if self.logger:
            self.logger.info(f"Proposed Action: {action['action']}")
            self.logger.info(f"Next State: {next_state['summary_state']}")
        else:
            logger.info(f"Proposed Action: {action['action']}")
            logger.info(f"Next State: {next_state['summary_state']}")
        print(
            f"==========Proposed Action and Next State from World Model of AgentLab Reasoners Agent=========="
        )
        print(f"Proposed Action: {action['action']}")
        print(f"Next State: {next_state['summary_state']}")

        next_state["memory"].update(state=state["summary_state"], intent=action["action"])
        next_state["memory"].step()

        return next_state, {"next_state": next_state}

    def is_terminal(self, state):
        # return False  # FIXME: this is a hack
        return state["step_idx"] > self.max_steps

    def update_example(self, example, **kwargs):
        super().update_example(example, **kwargs)


class SearchConfigWrapper(ReasonersSearchConfig):
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

        self.logger = None

    def get_actions(self, state):
        # Sample 20 actions
        llm_output = self.policy(
            state["summary_state"],
            state["memory"],
            llm_kwargs={
                "temperature": self.policy_temperature,
                "top_p": self.policy_top_p,
                "n": self.policy_n,
            },
            verbose=verbose_prompt,
        )

        action2freqs = {}
        for ans_dict in llm_output:  # structured to just be ans_dicts
            print(f"==========Policy Ans Dict from AgentLab Reasoners Agent==========")
            print(ans_dict)
            action = ans_dict.get("intent", None)
            if action is not None:
                freq, _ = action2freqs.get(action, (0, ""))
                action2freqs[action] = (freq + 1, ans_dict.get("think", None))

        if self.logger:
            self.logger.info(f"Action2Freqs: {action2freqs}")
        else:
            logger.info(f"Action2Freqs: {action2freqs}")
        # logger.info(f"Action2Freqs: {action2freqs}")

        cluster2freqs = {}
        while len(cluster2freqs) == 0:
            cluster2freqs = self._cluster_actions(action2freqs)
            if self.logger:
                self.logger.info(f"Cluster2Freqs: {cluster2freqs}")
                print(f"==========Cluster2Freqs from AgentLab Reasoners Agent==========")
                print(cluster2freqs)
            else:
                logger.info(f"Cluster2Freqs: {cluster2freqs}")

        action_freq_thoughts = [
            (action, freq, think) for action, (freq, think) in cluster2freqs.items()
        ]
        action_freq_thoughts.sort(key=lambda x: -x[1])
        action_freq_thoughts = action_freq_thoughts[: self.policy_freq_top_k]

        action_outputs = [
            {"action": action, "freq": freq, "think": think}
            for action, freq, think in action_freq_thoughts
        ]

        if self.logger:
            self.logger.info("Action Options:")
            for a in action_outputs:
                self.logger.info(f"Action: {a['action']}, Freq: {a['freq']}, Think: {a['think']}")
                print(f"==========Action Options from AgentLab Reasoners Agent==========")
                print(f"Action: {a['action']}, Freq: {a['freq']}, Think: {a['think']}")
        else:
            logger.info("Action Options:")
            for a in action_outputs:
                logger.info(f"Action: {a['action']}, Freq: {a['freq']}, Think: {a['think']}")

        return action_outputs

    def fast_reward(self, state, action):
        return 1.0, {}

    def reward(self, state, action, next_state, **kwargs):
        # ah critic is expected to evaluate many times
        # this is how reward is calculated. i see.
        llm_output = self.critic(
            next_state["summary_state"],
            next_state["memory"],
            llm_kwargs={
                "temperature": self.critic_temperature,
                "top_p": self.critic_top_p,
                "n": self.critic_n,
            },
            verbose=verbose_prompt,
        )

        """Assuming the following response format:
        Thoughts: <your thoughts and reasoning process>
        Status: "task_goal_reached" or "task_goal_not_reached"
        On the right track to success: “yes” or “no”
        """
        scores = []
        thoughts = []
        for ans_dict in llm_output:
            # logger.info(f"Critic Ans Dict: {ans_dict}")
            print(f"==========Critic Ans Dict from AgentLab Reasoners Agent==========")
            print(ans_dict)
            if ans_dict["status"] == "task_goal_reached":
                score = 100
            elif ans_dict["on_the_right_track"] == "yes":
                score = 1
            else:
                score = 0
            scores.append(score)
            thoughts.append(ans_dict.get("think"))
        reward = sum(scores) / len(scores)

        # logger.info(f"Score Examples: {scores[:5]}")
        # logger.info(f"Thought Examples: {thoughts[:5]}")
        if self.logger:
            self.logger.info(f"Thought Example: {thoughts[0]}")
            self.logger.info(f"Reward: {reward}")
        else:
            logger.info(f"Thought Example: {thoughts[0]}")
            logger.info(f"Reward: {reward}")

        print(f"==========Thought Example and Reward from AgentLab Reasoners Agent==========")
        print(f"Thought Example: {thoughts[0]}")
        print(f"Reward: {reward}")

        return reward, {"scores": scores, "thoughts": thoughts}

    def _cluster_actions(self, action2freqs):
        # TODO: Using the LLM Class to do this
        # ****API KEY NEEDED HERE****
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        action_candidate_dict = {i: action for i, action in enumerate(action2freqs.keys())}
        action_candidate_json = json.dumps(action_candidate_dict, indent=2)

        input_prompt = self._get_cluster_input_template().format(
            action_candidate_json=action_candidate_json
        )
        llm_prompt = (
            self._get_cluster_instruction_prompt()
            + "\n\n"
            + self._get_cluster_example_prompt()
            + "\n\n"
            + input_prompt
        )
        if verbose_prompt:
            print(f"==========Cluster Input Prompt from AgentLab Reasoners Agent==========")
            print(llm_prompt)

        # Run LLM for clustering
        system_prompt = "You are an expert at clustering text."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": llm_prompt},
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, response_format={"type": "json_object"}
        )
        response = completion.choices[0].message.content
        clusters_dict = json.loads(response)

        print(f"==========Cluster Results from AgentLab Reasoners Agent==========")
        print(clusters_dict)

        cluster2freqs = {}
        for cluster_id, cluster_info in clusters_dict.items():
            action = cluster_info.get("intent", None)
            if action is not None:
                cluster2freqs[action] = (0, "")
                for candidate_id in cluster_info.get("candidates", []):
                    candidate = action_candidate_dict.get(int(candidate_id), None)
                    if candidate is not None:
                        candidate_freq, candidate_think = action2freqs.get(candidate, (0, ""))
                cluster_freq, _ = cluster2freqs[action]
                cluster2freqs[action] = (cluster_freq + candidate_freq, candidate_think)
                # print(cluster2freqs)
        return cluster2freqs

    def _get_cluster_instruction_prompt(self):
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

    def _get_cluster_example_prompt(self):
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

    def _get_cluster_input_template(self):
        return """\
Concrete Example 2:

Dictionary of Intents:

{action_candidate_json}

Clustering Results:
"""

import copy
import json

from reasoners import SearchConfig as ReasonersSearchConfig
from reasoners import WorldModel as ReasonersWorldModel


class WorldModelWrapper(ReasonersWorldModel):
    def __init__(self, world_model, action_name, **kwargs):
        super().__init__()
        self.world_model = world_model
        self.action_name = action_name
        self.logger = None

    def init_state(self):
        return {
            'memory': copy.deepcopy(self.example['memory']),
            'state': self.example['state'],
            'action_history': [],
        }

    def step(self, state, action):
        """World Model"""
        # h[t+1] = f(h[t], s[t], a[t])
        next_memory = copy.deepcopy(state['memory'])
        memory_update = {
            'state': state['state'],
            self.action_name: action['action'],
            'plan': action['action'],
        }
        next_memory.update(**memory_update)

        llm_output = self.world_model(memory=next_memory, **next_memory.current_step)
        next_memory.step()

        next_state = {
            'state': llm_output['next_state'],
            'memory': next_memory,
            'action_history': state['action_history'] + [action['action']],
        }

        
        if self.logger:
            self.logger.info(f"Proposed Action: {action['action']}")
            if 'memory_update' in next_memory.history[-1]:
                memory_update = next_memory.history[-1]['memory_update']
                self.logger.info(f"Memory Update: {memory_update}")
            self.logger.info(f"Next State: {next_state['state']}")

        return next_state, {'next_state': next_state}

    def is_terminal(self, state):
        return False

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
        policy_output_name='action',
        critic_temperature=1.0,
        critic_top_p=0.95,
        critic_n=20,
        search_depth=1,
        llm_base_url=None,
        llm_api_key=None,
        **kwargs
    ):
        super().__init__()
        self.policy = policy
        self.critic = critic

        self.policy_temperature = policy_temperature
        self.policy_top_p = policy_top_p
        self.policy_n = policy_n
        self.policy_freq_top_k = policy_freq_top_k
        self.policy_output_name = policy_output_name

        self.critic_temperature = critic_temperature
        self.critic_top_p = critic_top_p
        self.critic_n = critic_n
        
        self.search_depth = search_depth
        
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key

        self.logger = None

    def get_actions(self, state):
        # Sample 20 actions
        llm_output = self.policy(
            state['state'],
            state['memory'],
            llm_kwargs={
                'temperature': self.policy_temperature,
                'top_p': self.policy_top_p,
                'n': self.policy_n,
            },
        )

        action2freqs = {}
        for ans_dict in llm_output['answers']:
            action = ans_dict[self.policy_output_name]
            freq, _ = action2freqs.get(action, (0, ''))
            action2freqs[action] = (freq + 1, ans_dict.get('think'))

        if self.logger:
            self.logger.info(f'Action2Freqs: {action2freqs}')

        cluster2freqs = {}
        while len(cluster2freqs) == 0:
            cluster2freqs = self._cluster_actions(action2freqs)
            if self.logger:
                self.logger.info(f'Cluster2Freqs: {cluster2freqs}')

        action_freq_thoughts = [
            (action, freq, think) for action, (freq, think) in cluster2freqs.items()
        ]
        action_freq_thoughts.sort(key=lambda x: -x[1])
        action_freq_thoughts = action_freq_thoughts[: self.policy_freq_top_k]

        action_outputs = [
            {'action': action, 'freq': freq, 'think': think}
            for action, freq, think in action_freq_thoughts
        ]

        if self.logger:
            self.logger.info(f'Num Actions Limit: {self.policy_freq_top_k}')
            self.logger.info('Action Options:')
            for a in action_outputs:
                self.logger.info(
                    f"Action: {a['action']}, Freq: {a['freq']}, Think: {a['think']}"
                )

        return action_outputs

    def fast_reward(self, state, action):
        return 0.0, {}

    def reward(self, state, action, next_state, **kwargs):
        depth = len(next_state['action_history'])
        if depth < self.search_depth:
            reward = 0.0
            if self.logger:
                self.logger.info(f'Current depth is {depth}, less than search depth {self.search_depth}')
                self.logger.info(f'Reward: {reward}')
            return reward, {}
        
        llm_output = self.critic(
            next_state['state'],
            next_state['memory'],
            llm_kwargs={
                'temperature': self.critic_temperature,
                'top_p': self.critic_top_p,
                'n': self.critic_n,
            },
        )
        answers = llm_output['answers']

        """Assuming the following response format:
        Thoughts: <your thoughts and reasoning process>
        Status: “success” or “failure”
        On the right track to success: “yes” or “no”
        """
        scores = []
        thoughts = []
        for ans_dict in answers:
            if ans_dict['status'].strip().strip('"').lower() == 'success':
                score = 1
            elif ans_dict['on_the_right_track'].strip().strip('"').lower() == 'yes':
                score = 0.5
            else:
                score = 0
            scores.append(score)
            thoughts.append(ans_dict.get('think'))
        reward = sum(scores) / len(scores)

        if self.logger:
            self.logger.info(f'Thought Example: {thoughts[0]}')
            self.logger.info(f'Num Reward Samples: {len(thoughts)}')
            self.logger.info(f'Reward: {reward}')


        return reward, {'scores': scores, 'thoughts': thoughts}

    def batch_reward(self, states, actions, next_states, **kwargs):
        batched_rewards, batched_data = [], []
        for state, action, next_state in zip(states, actions, next_states):
            reward, data = self.reward(state, action, next_state, **kwargs)
            batched_rewards.append(reward)
            batched_data.append(data)
        return batched_rewards, batched_data

    def _cluster_actions(self, action2freqs):
        action_candidate_dict = {
            i: action for i, action in enumerate(action2freqs.keys())
        }
        action_candidate_json = json.dumps(action_candidate_dict, indent=2)

        input_prompt = self._get_cluster_input_template().format(
            action_candidate_json=action_candidate_json
        )
        llm_prompt = (
            self._get_cluster_instruction_prompt()
            + '\n\n'
            + self._get_cluster_example_prompt()
            + '\n\n'
            + input_prompt
        )

        # Run LLM for clustering
        system_prompt = 'You are an expert at clustering text.'
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': llm_prompt},
        ]

        num_retries = 5
        for i in range(num_retries):
            try:
                cluster_llm = self.policy.llm
        
                response = cluster_llm.completion(
                    messages=messages,
                    response_format={'type': 'json_object'}
                )
                text = response['choices'][0]['message']['content'].strip()
                    
                self.logger.info(text)
                    
                clusters_dict = json.loads(text)
                break
            except Exception as e:
                if i == num_retries - 1:
                    raise e
                if self.logger:
                    self.logger.error(f'Error: {e}. Retrying...')

        cluster2freqs = {}
        for cluster_id, cluster_info in clusters_dict.items():
            action = cluster_info[self.policy_output_name]
            cluster2freqs[action] = (0, '')
            for candidate_id in cluster_info['candidates']:
                candidate = action_candidate_dict.get(int(candidate_id))
                if not candidate: # Skip if candidate is not found
                    continue
                candidate_freq, candidate_think = action2freqs.get(candidate, (0, ''))

                cluster_freq, _ = cluster2freqs[action]
                cluster2freqs[action] = (cluster_freq + candidate_freq, candidate_think)
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

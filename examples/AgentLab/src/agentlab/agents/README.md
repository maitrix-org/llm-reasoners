## Building LLM Agents in Your Library

This tutorial will guide you through the process of subclassing the `Agent` class to create agents that can interact with a `browsergym` environment. We'll cover the following steps:

1. Subclassing the `Agent` class
2. Implementing the `get_action` method
3. Customizing the `obs_preprocessor` method (optional)
4. Creating a compatible action set
5. Defining agent arguments
6. Running experiments with your agent

### Step 1: Subclassing the `Agent` Class

To create a custom agent, you need to subclass the `Agent` class and implement the abstract `get_action` method.

```python
from browsergym.experiment.loop import AbstractActionSet, DEFAULT_ACTION_SET
from browsergym.experiment.agent import Agent


class CustomAgent(Agent):
    def __init__(self):
        # define which action set your agent will be using
        self.action_set = DEFAULT_ACTION_SET

    def obs_preprocessor(self, obs: dict) -> Any:
        # Optionally override this method to customize observation preprocessing
        # The output of this method will be fed to the get_action method and also saved on disk.
        return super().obs_preprocessor(obs)

    def get_action(self, obs: Any) -> tuple[str, dict]:
        # Implement your custom logic here
        action = "your_action"
        info = {"custom_info": "details"}
        return action, info
```

### Step 2: Implementing the `get_action` Method

The `get_action` method updates the agent with the current observation and
returns the next action along with optional additional information i.e. all the
behavior of your agent goes here.

```python
def get_action(self, obs: dict) -> tuple[str, dict]:
    # Example implementation
    prompt = self.make_my_prompt_obs(obs)
    answer = self.llm(prompt)
    action, chain_of_thought = self.extract_action(answer)
    info = {
        "think": chain_of_thought,
        "messages": [prompt, answer],
        "stats": {"prompt_length": len(prompt), "answer_length": len(answer)},
        "some_other_info": "webagents are great",
    }
    return action, info
```

The info dictionnary is saved in the logs of the experiment. It can be used to
store any information you want to keep track of during the experiment. The keys
`"think"`, `"messages"`, and `"stats"` are reserved and will be displayed in AgentXray.

### Step 3: Customizing the `obs_preprocessor` Method (Optional)

The `obs_preprocessor` method preprocesses observations before they are fed to the `get_action` method. You can override this method to implement custom preprocessing logic.

```python
def obs_preprocessor(self, obs: dict) -> Any:
    # Example preprocessing logic
    obs["custom_key"] = "custom_value"
    return obs
```


### Step 4: Creating a Compatible Action Set

Your agent must use an action set that conforms to the `AbstractActionSet` class. The library provides a `HighLevelActionSet` with pre-implemented actions that you can use directly or customize.

```python
class CustomActionSet(AbstractActionSet):
    def describe(self, with_long_description: bool = True, with_examples: bool = True) -> str:
        return "Custom action set description."

    def example_action(self, abstract: bool) -> str:
        return "Example actions for in context learning."

    def to_python_code(self, action) -> str:
        return "executable python code"
```

### Step 5: Defining Agent Arguments

Define a class that inherits from `AgentArgs` to specify the arguments
required to instantiate your agent. This factory isolates all hyperparameters of
your agent and facilitate the experiment pipeline. Make sure it is a dataclass to
be compatible with the experiment pipeline. *As a requirement for dataclass, you
have to specify the type of each field (You can use Any if it is unknown)*

```python
from dataclasses import dataclass
from browsergym.experiment.agent import Agent
from browsergym.experiment.loop import AgentArgs


@dataclass
class CustomAgentArgs(AgentArgs):
    temperature: float = 0.5
    custom_param: str = "default_value"

    def make_agent(self) -> Agent:
        return CustomAgent(self.custom_param, self.temperature)
```

### Step 6: Running Experiments with Your Agent

To run experiments with your custom agent, define an instance of `ExpArgs` with the required parameters.

```python
from browsergym.experiment.loop import ExpArgs

exp_args = ExpArgs(
    agent_args=CustomAgentArgs(custom_param="value"),
    env_args=env_args,
    exp_dir="./experiments",
    exp_name="custom_experiment",
    enable_debug=True,
)

# Run the experiment
exp_args.prepare()
exp_args.run()
```

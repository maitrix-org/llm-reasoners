import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import bgym

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import make_system_message, make_user_message
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.llm_utils import (
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    extract_code_blocks,
    retry,
)
from agentlab.llm.tracking import cost_tracker_decorator

if TYPE_CHECKING:
    from agentlab.llm.chat_api import BaseModelArgs


@dataclass
class MostBasicAgentArgs(AgentArgs):
    agent_name: str = "BasicAgent"
    temperature: float = 0.1
    use_chain_of_thought: bool = False
    chat_model_args: "BaseModelArgs" = None

    def make_agent(self) -> bgym.Agent:
        return MostBasicAgent(
            temperature=self.temperature,
            use_chain_of_thought=self.use_chain_of_thought,
            chat_model_args=self.chat_model_args,
        )

    def set_reproducibility_mode(self):
        self.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()


class MostBasicAgent(bgym.Agent):
    def __init__(
        self, temperature: float, use_chain_of_thought: bool, chat_model_args: "BaseModelArgs"
    ):
        self.temperature = temperature
        self.use_chain_of_thought = use_chain_of_thought
        self.chat = chat_model_args.make_model()
        self.chat_model_args = chat_model_args

        self.action_set = bgym.HighLevelActionSet(["bid"], multiaction=False)

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str, dict]:
        messages = Discussion(SystemMessage("You are a web assistant."))
        messages.append(
            HumanMessage(
                f"""
You are helping a user to accomplish the following goal on a website:

{obs["goal"]}

To do so, you can interact with the environment using the following actions:

{self.action_set.describe(with_long_description=False)}

The inputs to those functions are the bids given in the html.

Here is the current state of the website, in the form of an html:

{obs["pruned_html"]}

The action you provide must be in between triple ticks and leverage the 'bid=' information provided in the html.
Here is an example of how to use the bid action:

```
click('a314')
```

Please provide a single action at a time and wait for the next observation. Provide only a single action per step. 
Focus on the bid that are given in the html, and use them to perform the actions.
"""
            )
        )
        if self.use_chain_of_thought:
            messages.add_text(
                f"""
Provide a chain of thoughts reasoning to decompose the task into smaller steps. And execute only the next step.
"""
            )

        def parser(response: str) -> tuple[dict, bool, str]:
            blocks = extract_code_blocks(response)
            if len(blocks) == 0:
                raise ParseError("No code block found in the response")
            action = blocks[0][1]
            thought = response
            return {"action": action, "think": thought}

        ans_dict = retry(self.chat, messages, n_retry=3, parser=parser)

        action = ans_dict.get("action", None)
        thought = ans_dict.get("think", None)

        return (
            action,
            bgym.AgentInfo(
                think=thought,
                chat_messages=messages,
                # put any stats that you care about as long as it is a number or a dict of numbers
                stats={"prompt_length": len(messages), "response_length": len(thought)},
                markdown_page="Add any txt information here, including base 64 images, to display in xray",
                extra_info={"chat_model_args": asdict(self.chat_model_args)},
            ),
        )


# example for a single task
env_args = bgym.EnvArgs(
    task_name="miniwob.click-button",
    task_seed=0,
    max_steps=10,
    headless=True,
)

chat_model_args = CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"]

# example for 2 experiments testing chain of thoughts on a miniwob task
exp_args = [
    bgym.ExpArgs(
        agent_args=MostBasicAgentArgs(
            temperature=0.1,
            use_chain_of_thought=True,
            chat_model_args=chat_model_args,
        ),
        env_args=env_args,
        logging_level=logging.INFO,
    ),
    bgym.ExpArgs(
        agent_args=MostBasicAgentArgs(
            temperature=0.1,
            use_chain_of_thought=False,
            chat_model_args=chat_model_args,
        ),
        env_args=env_args,
        logging_level=logging.INFO,
    ),
]

AGENT_4o_MINI = MostBasicAgentArgs(
    temperature=0.3,
    use_chain_of_thought=True,
    chat_model_args=chat_model_args,
)


def experiment_config():
    return exp_args

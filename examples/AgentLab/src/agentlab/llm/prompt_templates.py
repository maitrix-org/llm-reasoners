from dataclasses import dataclass
from typing import List

"""
To use this class, you should have the ``openai`` python package installed, and the
environment variable ``OPENAI_API_KEY`` set with your API key.
"""


@dataclass
class PromptTemplate:
    """
    Base class for prompt templates.

    Defines a standard interface for prompt templates, ensuring that they contain
    the required fields for the CustomLLMChatbot.
    """

    system: str
    human: str
    ai: str
    prompt_end: str = ""

    def format_message(self, message: dict) -> str:
        """
        Formats a given message based on its type.

        Args:
            message (dict): The message to be formatted.

        Returns:
            str: The formatted message.

        Raises:
            ValueError: If the message type is not supported.
        """
        if message["role"] == "system":
            return self.system.format(input=message["content"])
        elif message["role"] == "user":
            return self.human.format(input=message["content"])
        elif message["role"] == "assistant":
            return self.ai.format(input=message["content"])
        else:
            raise ValueError(f"Message role {message['role']} not supported")

    def construct_prompt(self, messages: List[dict]) -> str:
        """
        Constructs a prompt from a list of messages.

        Args:
            messages (List[BaseMessage]): The list of messages to be formatted.

        Returns:
            str: The constructed prompt.

        Raises:
            ValueError: If any element in the list is not of type BaseMessage.
        """
        if not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            raise ValueError("All elements in the list must be in openai format")

        prompt = "".join([self.format_message(m) for m in messages])
        prompt += self.prompt_end
        return prompt


def get_prompt_template(model_name):
    for key, value in MODEL_PREFIX_TO_PROMPT_TEMPLATES.items():
        if key in model_name:
            return value
    raise NotImplementedError(f"Model {model_name} has no supported chat template")


## Prompt templates

STARCHAT_PROMPT_TEMPLATE = PromptTemplate(
    system="<|system|>\n{input}<|end|>\n",
    human="<|user|>\n{input}<|end|>\n",
    ai="<|assistant|>\n{input}<|end|>\n",
    prompt_end="<|assistant|>",
)


## Model prefix to prompt template mapping

MODEL_PREFIX_TO_PROMPT_TEMPLATES = {
    "starcoder": STARCHAT_PROMPT_TEMPLATE,
    "starchat": STARCHAT_PROMPT_TEMPLATE,
}

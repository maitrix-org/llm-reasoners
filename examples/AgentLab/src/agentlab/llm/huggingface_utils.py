import logging
import time
from typing import Any, List, Optional

from pydantic import Field
from transformers import AutoTokenizer, GPT2TokenizerFast

from agentlab.llm.base_api import AbstractChatModel
from agentlab.llm.prompt_templates import PromptTemplate, get_prompt_template


class HFBaseChatModel(AbstractChatModel):
    """
    Custom LLM Chatbot that can interface with HuggingFace models.

    This class allows for the creation of a custom chatbot using models hosted
    on HuggingFace Hub or a local checkpoint. It provides flexibility in defining
    the temperature for response sampling and the maximum number of new tokens
    in the response.

    Attributes:
        llm (Any): The HuggingFaceHub model instance.
        prompt_template (Any): Template for the prompt to be used for the model's input sequence.
    """

    llm: Any = Field(description="The HuggingFaceHub model instance")
    tokenizer: Any = Field(
        default=None,
        description="The tokenizer to use for the model",
    )
    prompt_template: Optional[PromptTemplate] = Field(
        default=None,
        description="Template for the prompt to be used for the model's input sequence",
    )
    n_retry_server: int = Field(
        default=4,
        description="The number of times to retry the server if it fails to respond",
    )

    def __init__(self, model_name, n_retry_server):
        super().__init__()
        self.n_retry_server = n_retry_server

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if isinstance(self.tokenizer, GPT2TokenizerFast):
            logging.warning(
                f"No chat template is defined for {model_name}. Resolving to the hard-coded templates."
            )
            self.tokenizer = None
            self.prompt_template = get_prompt_template(model_name)

    def __call__(
        self,
        messages: list[dict],
    ) -> dict:

        # NOTE: The `stop`, `run_manager`, and `kwargs` arguments are ignored in this implementation.

        if self.tokenizer:
            # messages_formated = _convert_messages_to_dict(messages) ## ?
            try:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as e:
                if "Conversation roles must alternate" in str(e):
                    logging.warning(
                        f"Failed to apply the chat template. Maybe because it doesn't support the 'system' role"
                        "Retrying with the 'system' role appended to the 'user' role."
                    )
                    messages = _prepend_system_to_first_user(messages)
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    raise e

        elif self.prompt_template:
            prompt = self.prompt_template.construct_prompt(messages)

        itr = 0
        while True:
            try:
                response = self.llm(prompt)
                return response
            except Exception as e:
                if itr == self.n_retry_server - 1:
                    raise e
                logging.warning(
                    f"Failed to get a response from the server: \n{e}\n"
                    f"Retrying... ({itr+1}/{self.n_retry_server})"
                )
                time.sleep(5)
                itr += 1

    def _llm_type(self):
        return "huggingface"


def _prepend_system_to_first_user(messages, column_remap={}):
    # Initialize an index for the system message
    system_index = None

    human_key = column_remap.get("HumanMessage", "user")
    role_key = column_remap.get("role", "role")
    text_key = column_remap.get("text", "content")

    # Find the system content and its index
    for i, msg in enumerate(messages):
        if msg[role_key] == "system":
            system_index = i
            system_content = msg[text_key]
            break  # Stop after finding the first system message

    # If a system message was found, modify the first user message and remove the system message
    if system_index is not None:
        for msg in messages:
            if msg[role_key] == human_key:
                # Prepend system content to the first user content
                msg[text_key] = system_content + "\n" + msg[text_key]
                # Remove the original system message
                del messages[system_index]
                break  # Ensures that only the first user message is modified

    return messages

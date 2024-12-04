import base64
import collections
import io
import json
import logging
import os
import re
import time
from copy import deepcopy
from functools import cache
from typing import TYPE_CHECKING, Any, Union
from warnings import warn

import numpy as np
import tiktoken
import yaml
from langchain.schema import BaseMessage
from langchain_community.adapters.openai import convert_message_to_dict
from PIL import Image
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from agentlab.llm.chat_api import ChatModel


def messages_to_dict(messages: list[dict] | list[BaseMessage]) -> dict:
    new_messages = Discussion()
    for m in messages:
        if isinstance(m, dict):
            new_messages.add_message(m)
        elif isinstance(m, str):
            new_messages.add_message({"role": "<unknown role>", "content": m})
        elif isinstance(m, BaseMessage):
            new_messages.add_message(convert_message_to_dict(m))
        else:
            raise ValueError(f"Unknown message type: {type(m)}")
    return new_messages


class RetryError(ValueError):
    pass


def retry(
    chat: "ChatModel",
    messages: "Discussion",
    n_retry: int,
    parser: callable,
    log: bool = True,
):
    """Retry querying the chat models with the response from the parser until it
    returns a valid value.

    If the answer is not valid, it will retry and append to the chat the  retry
    message.  It will stop after `n_retry`.

    Note, each retry has to resend the whole prompt to the API. This can be slow
    and expensive.

    Args:
        chat (ChatModel): a ChatModel object taking a list of messages and
            returning a list of answers, all in OpenAI format.
        messages (list): the list of messages so far. This list will be modified with
            the new messages and the retry messages.
        n_retry (int): the maximum number of sequential retries.
        parser (callable): a function taking a message and retruning a parsed value,
            or raising a ParseError
        log (bool): whether to log the retry messages.

    Returns:
        dict: the parsed value, with a string at key "action".

    Raises:
        ParseError: if the parser could not parse the response after n_retry retries.
    """
    tries = 0
    while tries < n_retry:
        answer = chat(messages)
        # TODO: could we change this to not use inplace modifications ?
        messages.append(answer)
        try:
            return parser(answer["content"])
        except ParseError as parsing_error:
            tries += 1
            if log:
                msg = f"Query failed. Retrying {tries}/{n_retry}.\n[LLM]:\n{answer['content']}\n[User]:\n{str(parsing_error)}"
                logging.info(msg)
            messages.append(dict(role="user", content=str(parsing_error)))

    raise ParseError(f"Could not parse a valid value after {n_retry} retries.")


def truncate_tokens(text, max_tokens=8000, start=0, model_name="gpt-4"):
    """Use tiktoken to truncate a text to a maximum number of tokens."""
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    if len(tokens) - start > max_tokens:
        return enc.decode(tokens[start : (start + max_tokens)])
    else:
        return text


@cache
def get_tokenizer_old(model_name="openai/gpt-4"):
    if model_name.startswith("test"):
        return tiktoken.encoding_for_model("gpt-4")
    if model_name.startswith("openai"):
        return tiktoken.encoding_for_model(model_name.split("/")[-1])
    if model_name.startswith("azure"):
        return tiktoken.encoding_for_model(model_name.split("/")[1])
    if model_name.startswith("reka"):
        logging.warning(
            "Reka models don't have a tokenizer implemented yet. Using the default one."
        )
        return tiktoken.encoding_for_model("gpt-4")
    else:
        return AutoTokenizer.from_pretrained(model_name)


@cache
def get_tokenizer(model_name="gpt-4"):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logging.info(f"Could not find a tokenizer for model {model_name}. Trying HuggingFace.")
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except OSError:
        logging.info(f"Could not find a tokenizer for model {model_name}. Defaulting to gpt-4.")
    return tiktoken.encoding_for_model("gpt-4")


def count_tokens(text, model="openai/gpt-4"):
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def json_parser(message):
    """Parse a json message for the retry function."""

    try:
        value = json.loads(message)
        valid = True
        retry_message = ""
    except json.JSONDecodeError as e:
        warn(e)
        value = {}
        valid = False
        retry_message = "Your response is not a valid json. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def yaml_parser(message):
    """Parse a yaml message for the retry function."""

    # saves gpt-3.5 from some yaml parsing errors
    message = re.sub(r":\s*\n(?=\S|\n)", ": ", message)

    try:
        value = yaml.safe_load(message)
        valid = True
        retry_message = ""
    except yaml.YAMLError as e:
        warn(str(e))
        value = {}
        valid = False
        retry_message = "Your response is not a valid yaml. Please try again and be careful to the format. Don't add any apology or comment, just the answer."
    return value, valid, retry_message


def _compress_chunks(text, identifier, skip_list, split_regex="\n\n+"):
    """Compress a string by replacing redundant chunks by identifiers. Chunks are defined by the split_regex."""
    text_list = re.split(split_regex, text)
    text_list = [chunk.strip() for chunk in text_list]
    counter = collections.Counter(text_list)
    def_dict = {}
    id = 0

    # Store items that occur more than once in a dictionary
    for item, count in counter.items():
        if count > 1 and item not in skip_list and len(item) > 10:
            def_dict[f"{identifier}-{id}"] = item
            id += 1

    # Replace redundant items with their identifiers in the text
    compressed_text = "\n".join(text_list)
    for key, value in def_dict.items():
        compressed_text = compressed_text.replace(value, key)

    return def_dict, compressed_text


def compress_string(text):
    """Compress a string by replacing redundant paragraphs and lines with identifiers."""

    # Perform paragraph-level compression
    def_dict, compressed_text = _compress_chunks(
        text, identifier="§", skip_list=[], split_regex="\n\n+"
    )

    # Perform line-level compression, skipping any paragraph identifiers
    line_dict, compressed_text = _compress_chunks(
        compressed_text, "¶", list(def_dict.keys()), split_regex="\n+"
    )
    def_dict.update(line_dict)

    # Create a definitions section
    def_lines = ["<definitions>"]
    for key, value in def_dict.items():
        def_lines.append(f"{key}:\n{value}")
    def_lines.append("</definitions>")
    definitions = "\n".join(def_lines)

    return definitions + "\n" + compressed_text


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    All text and keys will be converted to lowercase before matching.

    Args:
        text (str): The input string containing the HTML tags.
        keys (list[str]): The HTML tags to extract the content from.

    Returns:
        dict: A dictionary mapping each key to a list of subset in `text` that match the key.
    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


class ParseError(Exception):
    pass


def extract_code_blocks(text) -> list[tuple[str, str]]:
    pattern = re.compile(r"```(\w*\n)?(.*?)```", re.DOTALL)

    matches = pattern.findall(text)
    return [(match[0].strip(), match[1].strip()) for match in matches]


def parse_html_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_html_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_html_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Args:
        text (str): The input string containing the HTML tags.
        keys (list[str]): The HTML tags to extract the content from.
        optional_keys (list[str]): The HTML tags to extract the content from, but are optional.
        merge_multiple (bool): Whether to merge multiple instances of the same key.

    Returns:
        dict: A dictionary mapping each key to a subset of `text` that match the key.
        bool: Whether the parsing was successful.
        str: A message to be displayed to the agent if the parsing was not successful.

    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if not key in content_dict:
            if not key in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


def download_and_save_model(model_name: str, save_dir: str = "."):
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model downloaded and saved to {save_dir}")


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


class BaseMessage(dict):
    def __init__(self, role: str, content: Union[str, list[dict]]):
        self["role"] = role
        self["content"] = deepcopy(content)

    def __str__(self, warn_if_image=False) -> str:
        if isinstance(self["content"], str):
            return self["content"]
        if not all(elem["type"] == "text" for elem in self["content"]):
            msg = "The content of the message has images, which are not displayed in the string representation."
            if warn_if_image:
                logging.warning(msg)
            else:
                logging.info(msg)

        return "\n".join([elem["text"] for elem in self["content"] if elem["type"] == "text"])

    def add_content(self, type: str, content: Any):
        if isinstance(self["content"], str):
            text = self["content"]
            self["content"] = []
            self["content"].append({"type": "text", "text": text})
        self["content"].append({"type": type, type: content})

    def add_text(self, text: str):
        self.add_content("text", text)

    def add_image(self, image: np.ndarray | Image.Image | str, detail: str = None):
        if not isinstance(image, str):
            image_url = image_to_jpg_base64_url(image)
        else:
            image_url = image
        if detail:
            self.add_content("image_url", {"url": image_url, "detail": detail})
        else:
            self.add_content("image_url", {"url": image_url})

    def to_markdown(self):
        if isinstance(self["content"], str):
            return f"\n```\n{self['content']}\n```\n"
        res = []
        for elem in self["content"]:
            # add texts between ticks and images
            if elem["type"] == "text":
                res.append(f"\n```\n{elem['text']}\n```\n")
            elif elem["type"] == "image_url":
                img_str = (
                    elem["image_url"]
                    if isinstance(elem["image_url"], str)
                    else elem["image_url"]["url"]
                )
                res.append(f"![image]({img_str})")
        return "\n".join(res)

    def merge(self):
        """Merges content elements of type 'text' if they are adjacent."""
        if isinstance(self["content"], str):
            return
        new_content = []
        for elem in self["content"]:
            if elem["type"] == "text":
                if new_content and new_content[-1]["type"] == "text":
                    new_content[-1]["text"] += "\n" + elem["text"]
                else:
                    new_content.append(elem)
            else:
                new_content.append(elem)
        self["content"] = new_content


class SystemMessage(BaseMessage):
    def __init__(self, content: Union[str, list[dict]]):
        super().__init__("system", content)


class HumanMessage(BaseMessage):
    def __init__(self, content: Union[str, list[dict]]):
        super().__init__("user", content)


class AIMessage(BaseMessage):
    def __init__(self, content: Union[str, list[dict]]):
        super().__init__("assistant", content)


class Discussion:
    def __init__(self, messages: Union[list[BaseMessage], BaseMessage] = None):
        if isinstance(messages, BaseMessage):
            messages = [messages]
        elif messages is None:
            messages = []
        self.messages = messages

    @property
    def last_message(self):
        return self.messages[-1]

    def merge(self):
        for m in self.messages:
            m.merge()

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.messages)

    def to_string(self):
        self.merge()
        return str(self)

    def to_openai(self):
        self.merge()
        return self.messages

    def add_message(
        self,
        message: BaseMessage | dict = None,
        role: str = None,
        content: Union[str, list[dict]] = None,
    ):
        if message is None:
            message = BaseMessage(role, content)
        else:
            if isinstance(message, dict):
                message = BaseMessage(**message)
        self.messages.append(message)

    def append(self, message: BaseMessage | dict):
        self.add_message(message)

    def add_content(self, type: str, content: Any):
        """Add content to the last message."""
        self.last_message.add_content(type, content)

    def add_text(self, text: str):
        """Add text to the last message."""
        self.last_message.add_text(text)

    def add_image(self, image: np.ndarray | Image.Image | str, detail: str = None):
        """Add an image to the last message."""
        self.last_message.add_image(image, detail)

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, key):
        return self.messages[key]

    def to_markdown(self):
        self.merge()
        return "\n".join([f"Message {i}\n{m.to_markdown()}\n" for i, m in enumerate(self.messages)])


if __name__ == "__main__":

    # model_to_download = "THUDM/agentlm-70b"
    model_to_download = "databricks/dbrx-instruct"
    save_dir = "/mnt/ui_copilot/data_rw/base_models/"
    # set the following env variable to enable the transfer of the model
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    download_and_save_model(model_to_download, save_dir=save_dir)

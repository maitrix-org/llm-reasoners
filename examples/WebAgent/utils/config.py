# import argparse
import logging
import os
# import pathlib
# import platform
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from types import UnionType
from typing import Any, ClassVar, get_args, get_origin

# import toml
from dotenv import load_dotenv

from .singleton import Singleton

logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class LLMConfig(metaclass=Singleton):
    """
    Configuration for the LLM model.

    Attributes:
        model: The model to use.
        api_key: The API key to use.
        base_url: The base URL for the API. This is necessary for local LLMs. It is also used for Azure embeddings.
        api_version: The version of the API.
        embedding_model: The embedding model to use.
        embedding_base_url: The base URL for the embedding API.
        embedding_deployment_name: The name of the deployment for the embedding API. This is used for Azure OpenAI.
        aws_access_key_id: The AWS access key ID.
        aws_secret_access_key: The AWS secret access key.
        aws_region_name: The AWS region name.
        num_retries: The number of retries to attempt.
        retry_min_wait: The minimum time to wait between retries, in seconds. This is exponential backoff minimum. For models with very low limits, this can be set to 15-20.
        retry_max_wait: The maximum time to wait between retries, in seconds. This is exponential backoff maximum.
        timeout: The timeout for the API.
        max_chars: The maximum number of characters to send to and receive from the API. This is a fallback for token counting, which doesn't work in all cases.
        temperature: The temperature for the API.
        top_p: The top p for the API.
        custom_llm_provider: The custom LLM provider to use. This is undocumented in opendevin, and normally not used. It is documented on the litellm side.
        max_input_tokens: The maximum number of input tokens. Note that this is currently unused, and the value at runtime is actually the total tokens in OpenAI (e.g. 128,000 tokens for GPT-4).
        max_output_tokens: The maximum number of output tokens. This is sent to the LLM.
        input_cost_per_token: The cost per input token. This will available in logs for the user to check.
        output_cost_per_token: The cost per output token. This will available in logs for the user to check.
    """

    model: str = 'gpt-4o'
    api_key: str | None = None
    base_url: str | None = None
    model_port_config_file: str | None = None
    api_version: str | None = None
    embedding_model: str = 'local'
    embedding_base_url: str | None = None
    embedding_deployment_name: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region_name: str | None = None
    num_retries: int = 5
    retry_min_wait: int = 3
    retry_max_wait: int = 60
    timeout: int | None = None
    max_chars: int = 5_000_000  # fallback for token counting
    temperature: float = 0
    top_p: float = 0.5
    custom_llm_provider: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None

    def defaults_to_dict(self) -> dict:
        """
        Serialize fields to a dict for the frontend, including type hints, defaults, and whether it's optional.
        """
        dict = {}
        for f in fields(self):
            dict[f.name] = get_field_info(f)
        return dict

    def __str__(self):
        attr_str = []
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, f.name)

            if attr_name in ['api_key', 'aws_access_key_id', 'aws_secret_access_key']:
                attr_value = '******' if attr_value else None

            attr_str.append(f'{attr_name}={repr(attr_value)}')

        return f"LLMConfig({', '.join(attr_str)})"

    def __repr__(self):
        return self.__str__()

    
@dataclass
class AgentConfig(metaclass=Singleton):
    """
    Configuration for the agent.

    Attributes:
        name: The name of the agent.
        memory_enabled: Whether long-term memory (embeddings) is enabled.
        memory_max_threads: The maximum number of threads indexing at the same time for embeddings.
    """

    name: str = 'CodeActAgent'
    memory_enabled: bool = False
    memory_max_threads: int = 2

    def defaults_to_dict(self) -> dict:
        """
        Serialize fields to a dict for the frontend, including type hints, defaults, and whether it's optional.
        """
        dict = {}
        for f in fields(self):
            dict[f.name] = get_field_info(f)
        return dict
    
    
@dataclass
class AppConfig(metaclass=Singleton):
    """
    Configuration for the app.

    Attributes:
        llm: The LLM configuration.
        agent: The agent configuration.
        runtime: The runtime environment.
        file_store: The file store to use.
        file_store_path: The path to the file store.
        workspace_base: The base path for the workspace. Defaults to ./workspace as an absolute path.
        workspace_mount_path: The path to mount the workspace. This is set to the workspace base by default.
        workspace_mount_path_in_sandbox: The path to mount the workspace in the sandbox. Defaults to /workspace.
        workspace_mount_rewrite: The path to rewrite the workspace mount path to.
        cache_dir: The path to the cache directory. Defaults to /tmp/cache.
        sandbox_container_image: The container image to use for the sandbox.
        run_as_devin: Whether to run as devin.
        max_iterations: The maximum number of iterations.
        max_budget_per_task: The maximum budget allowed per task, beyond which the agent will stop.
        e2b_api_key: The E2B API key.
        sandbox_type: The type of sandbox to use. Options are: ssh, exec, e2b, local.
        use_host_network: Whether to use the host network.
        ssh_hostname: The SSH hostname.
        disable_color: Whether to disable color. For terminals that don't support color.
        sandbox_user_id: The user ID for the sandbox.
        sandbox_timeout: The timeout for the sandbox.
        debug: Whether to enable debugging.
        enable_auto_lint: Whether to enable auto linting. This is False by default, for regular runs of the app. For evaluation, please set this to True.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    runtime: str = 'server'
    file_store: str = 'memory'
    file_store_path: str = '/tmp/file_store'
    workspace_base: str = os.path.join(os.getcwd(), 'workspace')
    workspace_mount_path: str | None = None
    workspace_mount_path_in_sandbox: str = '/workspace'
    workspace_mount_rewrite: str | None = None
    cache_dir: str = '/tmp/cache'
    sandbox_container_image: str = 'ghcr.io/opendevin/sandbox' + (
        f':{os.getenv("OPEN_DEVIN_BUILD_VERSION")}'
        if os.getenv('OPEN_DEVIN_BUILD_VERSION')
        else ':main'
    )
    run_as_devin: bool = True
    max_iterations: int = 100
    max_budget_per_task: float | None = None
    e2b_api_key: str = ''
    sandbox_type: str = 'ssh'  # Can be 'ssh', 'exec', or 'e2b'
    use_host_network: bool = False
    ssh_hostname: str = 'localhost'
    disable_color: bool = False
    sandbox_user_id: int = os.getuid() if hasattr(os, 'getuid') else 1000
    sandbox_timeout: int = 120
    initialize_plugins: bool = True
    persist_sandbox: bool = False
    ssh_port: int = 63710
    ssh_password: str | None = None
    jwt_secret: str = uuid.uuid4().hex
    debug: bool = False
    enable_auto_lint: bool = (
        False  # once enabled, OpenDevin would lint files after editing
    )

    defaults_dict: ClassVar[dict] = {}

    def __post_init__(self):
        """
        Post-initialization hook, called when the instance is created with only default values.
        """
        AppConfig.defaults_dict = self.defaults_to_dict()

    def defaults_to_dict(self) -> dict:
        """
        Serialize fields to a dict for the frontend, including type hints, defaults, and whether it's optional.
        """
        dict = {}
        for f in fields(self):
            field_value = getattr(self, f.name)

            # dataclasses compute their defaults themselves
            if is_dataclass(type(field_value)):
                dict[f.name] = field_value.defaults_to_dict()
            else:
                dict[f.name] = get_field_info(f)
        return dict

    def __str__(self):
        attr_str = []
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, f.name)

            if attr_name in ['e2b_api_key', 'github_token']:
                attr_value = '******' if attr_value else None

            attr_str.append(f'{attr_name}={repr(attr_value)}')

        return f"AppConfig({', '.join(attr_str)}"

    def __repr__(self):
        return self.__str__()
    
    
def get_field_info(field):
    """
    Extract information about a dataclass field: type, optional, and default.

    Args:
        field: The field to extract information from.

    Returns: A dict with the field's type, whether it's optional, and its default value.
    """
    field_type = field.type
    optional = False

    # for types like str | None, find the non-None type and set optional to True
    # this is useful for the frontend to know if a field is optional
    # and to show the correct type in the UI
    # Note: this only works for UnionTypes with None as one of the types
    if get_origin(field_type) is UnionType:
        types = get_args(field_type)
        non_none_arg = next((t for t in types if t is not type(None)), None)
        if non_none_arg is not None:
            field_type = non_none_arg
            optional = True

    # type name in a pretty format
    type_name = (
        field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
    )

    # default is always present
    default = field.default

    # return a schema with the useful info for frontend
    return {'type': type_name.lower(), 'optional': optional, 'default': default}


# llm_config = LLMConfig()

config = AppConfig()
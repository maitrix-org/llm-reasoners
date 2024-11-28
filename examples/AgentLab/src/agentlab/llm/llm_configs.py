from agentlab.llm.chat_api import (
    AzureModelArgs,
    OpenAIModelArgs,
    OpenRouterModelArgs,
    SelfHostedModelArgs,
)

default_oss_llms_args = {
    "n_retry_server": 4,
    "temperature": 0.01,
}

CLOSED_SOURCE_APIS = [
    "openai",
    "reka",
    "test",
]

CHAT_MODEL_ARGS_DICT = {
    "openai/gpt-4o-mini-2024-07-18": OpenAIModelArgs(
        model_name="gpt-4o-mini-2024-07-18",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "openai/gpt-4-1106-preview": OpenAIModelArgs(
        model_name="gpt-4-1106-preview",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=4_096,
    ),
    "openai/gpt-4-vision-preview": OpenAIModelArgs(
        model_name="gpt-4-vision-preview",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-4o-2024-05-13": OpenAIModelArgs(
        model_name="gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=4_096,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),
    "openai/gpt-3.5-turbo-0125": OpenAIModelArgs(
        model_name="gpt-3.5-turbo-0125",
        max_total_tokens=16_384,
        max_input_tokens=16_384,
        max_new_tokens=4096,
    ),
    "openai/gpt-3.5-turbo-1106": OpenAIModelArgs(
        model_name="gpt-3.5-turbo-1106",
        max_total_tokens=16_384,
        max_input_tokens=16_384,
        max_new_tokens=4096,
    ),
    "azure/gpt-35-turbo/gpt-35-turbo": AzureModelArgs(
        model_name="gpt-35-turbo",
        deployment_name="gpt-35-turbo",
        max_total_tokens=8_192,
        max_input_tokens=7500,
        max_new_tokens=500,
    ),
    "azure/gpt-4o-2024-05-13": AzureModelArgs(
        model_name="gpt-4o",
        deployment_name="gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-4o-2024-08-06": AzureModelArgs(
        model_name="gpt-4o",
        deployment_name="gpt-4o-2024-08-06",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    "azure/gpt-4o-mini-2024-07-18": AzureModelArgs(
        model_name="gpt-4o-mini",
        deployment_name="gpt-4o-mini-2024-07-18",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=16_384,
        vision_support=True,
    ),
    # ---------------- OSS LLMs ----------------#
    "meta-llama/Meta-Llama-3-70B-Instruct": SelfHostedModelArgs(
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        max_total_tokens=8_192,
        max_input_tokens=8_192 - 512,
        max_new_tokens=512,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": SelfHostedModelArgs(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        max_total_tokens=16_384,
        max_input_tokens=16_384 - 512,
        max_new_tokens=512,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": SelfHostedModelArgs(
        model_name="mistralai/Mixtral-8x22B-Instruct-v0.1",
        max_total_tokens=32_000,
        max_input_tokens=30_000,
        max_new_tokens=2_000,
        backend="huggingface",
        **default_oss_llms_args,
    ),
    # ---------------- OPENROUTER ----------------#
    "openrouter/meta-llama/llama-3.1-405b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-405b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-70b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-70b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3-70b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3-70b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-8b-instruct:free": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-8b-instruct:free",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/meta-llama/llama-3.1-8b-instruct": OpenRouterModelArgs(
        model_name="meta-llama/llama-3.1-8b-instruct",
        max_total_tokens=128_000,
        max_input_tokens=100_000,
        max_new_tokens=28_000,
        temperature=1e-1,
    ),
    "openrouter/anthropic/claude-3.5-sonnet:beta": OpenRouterModelArgs(
        model_name="anthropic/claude-3.5-sonnet:beta",
        max_total_tokens=200_000,
        max_input_tokens=200_000,
        max_new_tokens=8_192,
        temperature=1e-1,
        vision_support=True,
    ),
    "openrouter/qwen/qwen-2-72b-instruct": OpenRouterModelArgs(
        model_name="qwen/qwen-2-72b-instruct",
        max_total_tokens=32_000,
        max_input_tokens=30_000,
        max_new_tokens=2_000,
        temperature=1e-1,
    ),
    "openrouter/openai/o1-mini-2024-09-12": OpenRouterModelArgs(
        model_name="openai/o1-mini-2024-09-12",
        max_total_tokens=128_000,
        max_input_tokens=128_000,
        max_new_tokens=64_000,
        temperature=1e-1,
    ),
}

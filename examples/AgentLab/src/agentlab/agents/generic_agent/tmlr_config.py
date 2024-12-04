from copy import deepcopy

from agentlab.agents import dynamic_prompting as dp
from agentlab.experiments import args
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .generic_agent import GenericAgentArgs
from .generic_agent_prompt import GenericPromptFlags

BASE_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,  # gpt-4o config except for this line
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        multi_actions=False,
        action_set="bid",
        long_description=False,
        individual_examples=False,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=None,
)


def get_base_agent(llm_config: str):
    return GenericAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[llm_config],
        flags=BASE_FLAGS,
    )


def get_vision_agent(llm_config: str):
    flags = deepcopy(BASE_FLAGS)
    flags.obs.use_screenshot = True
    agent_args = GenericAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[llm_config],
        flags=flags,
    )
    agent_args.agent_name = f"{agent_args.agent_name}_vision"
    return agent_args


def get_som_agent(llm_config: str):
    flags = deepcopy(BASE_FLAGS)
    flags.obs.use_screenshot = True
    flags.obs.use_som = True
    return GenericAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT[llm_config],
        flags=flags,
    )

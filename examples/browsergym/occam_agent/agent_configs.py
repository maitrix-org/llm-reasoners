import bgym

from agentlab.agents import dynamic_prompting as dp
from agentlab.experiments import args
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .occam_agent import OccamAgentArgs
from .occam_agent_prompt import OccamPromptFlags

# GPT-4o default config
FLAGS_GPT_4o = OccamPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
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
        action_set=bgym.HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
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

AGENT_4o = OccamAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=FLAGS_GPT_4o,
)

AGENT_4o_MINI = OccamAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    flags=FLAGS_GPT_4o,
)

# GPT-4o vision default config
FLAGS_GPT_4o_VISION = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o_VISION.obs.use_screenshot = True
FLAGS_GPT_4o_VISION.obs.use_som = True

AGENT_4o_VISION = OccamAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
    flags=FLAGS_GPT_4o_VISION,
)

FLAGS_DEEPSEEK = OccamPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=False,
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
        action_set=bgym.HighLevelActionSetArgs(
            subsets=["occam"],
            multiaction=False,
        ),
        long_description=True,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=True,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=None,
)

AGENT_DEEPSEEK = OccamAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["deepseek"],
    flags=FLAGS_DEEPSEEK,
)

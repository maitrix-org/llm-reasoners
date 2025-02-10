import bgym
from copy import deepcopy

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs
from agentlab.experiments import args
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from .agent import PlanAgentArgs
from .planner import (
    PlannerArgs,
    PlannerLLMArgs,
    SearchAlgorithmArgs,
    WorldModelArgs,
    CriticArgs,
    PolicyArgs,
)
from .prompts import PlanAgentPromptFlags


# GPT-4o default config
FLAGS_GPT_4o = PlanAgentPromptFlags(
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
    use_plan=True,
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

BASELINE_AGENT_FLAGS = deepcopy(FLAGS_GPT_4o)
BASELINE_AGENT_FLAGS.use_plan = False
BASELINE_AGENT_4o_MINI = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    flags=BASELINE_AGENT_FLAGS,
)

PLAN_AGENT_4o_MINI = PlanAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    flags=FLAGS_GPT_4o,
    planner_args=PlannerArgs(
        llm_args=PlannerLLMArgs(
            model="gpt-4o-mini-2024-07-18",
            max_tokens=16_384,
            temperature=0.7,
            additional_prompt=None,
            backend="openai",
            is_instruct_model=True,
        ),
        search_algorithm_args=SearchAlgorithmArgs(
            algorithm="MCTS", n_iters=3, depth_limit=2, w_exp=10
        ),
        world_model_args=WorldModelArgs(mode="default"),
        critic_args=CriticArgs(mode="default"),
        policy_args=PolicyArgs(mode="default"),
    ),
    action_set_args=bgym.HighLevelActionSetArgs(
        subsets=["chat", "bid", "nav"],
        strict=False,
        multiaction=False,
    ),
)


# GPT-4o vision default config
FLAGS_GPT_4o_VISION = FLAGS_GPT_4o.copy()
FLAGS_GPT_4o_VISION.obs.use_screenshot = True
FLAGS_GPT_4o_VISION.obs.use_som = True

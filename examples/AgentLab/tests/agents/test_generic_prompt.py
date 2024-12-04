from copy import deepcopy

import bgym
import pytest

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.agents.generic_agent.generic_agent_prompt import (
    GenericPromptFlags,
    MainPrompt,
)
from agentlab.llm.llm_utils import count_tokens

html_template = """
<html>
<body>
<div>
Hello World.
Step {}.
</div>
</body>
some extra text to make the html longer
</html>
"""

base_obs = {
    "goal": "do this and that",
    "goal_object": [{"type": "text", "text": "do this and that"}],
    "chat_messages": [{"role": "user", "message": "do this and that"}],
    "axtree_txt": "[1] Click me",
    "focused_element_bid": "45-256",
    "open_pages_urls": ["https://example.com"],
    "open_pages_titles": ["Example"],
    "active_page_index": 0,
}

OBS_HISTORY = [
    base_obs
    | {
        "pruned_html": html_template.format(1),
        "last_action_error": "",
    },
    base_obs
    | {
        "pruned_html": html_template.format(2),
        "last_action_error": "Hey, this is an error in the past",
    },
    base_obs
    | {
        "pruned_html": html_template.format(3),
        "last_action_error": "Hey, there is an error now",
    },
]
ACTIONS = ["click('41')", "click('42')"]
MEMORIES = ["memory A", "memory B"]
THOUGHTS = ["thought A", "thought B"]

ALL_TRUE_FLAGS = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=True,
        use_ax_tree=True,
        use_tabs=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=True,
        use_action_history=True,
        use_think_history=True,
        use_diff=True,
        html_type="pruned_html",
        use_screenshot=False,  # TODO test this
        use_som=False,  # TODO test this
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords=False,
        filter_visible_elements_only=True,
    ),
    action=dp.ActionFlags(
        action_set=bgym.HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=True,
        ),
        long_description=True,
        individual_examples=True,
    ),
    use_plan=True,
    use_criticise=True,
    use_thinking=True,
    use_memory=True,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,  # TODO test this
    max_prompt_tokens=None,
    be_cautious=True,
    extra_instructions=None,
)


FLAG_EXPECTED_PROMPT = [
    (
        "obs.use_html",
        ("HTML:", "</html>", "Hello World.", "Step 3."),  # last obs will be in obs
    ),
    (
        "obs.use_ax_tree",
        ("AXTree:", "Click me"),
    ),
    (
        "obs.use_tabs",
        ("Currently open tabs:", "(active tab)"),
    ),
    (
        "obs.use_focused_element",
        ("Focused element:", "bid='45-256'"),
    ),
    (
        "obs.use_error_logs",
        ("Hey, there is an error now",),
    ),
    (
        "use_plan",
        ("You just executed step", "1- think\n2- do it"),
    ),
    (
        "use_criticise",
        (
            "Criticise action_draft",
            "<criticise>",
            "</criticise>",
            "<action_draft>",
        ),
    ),
    (
        "use_thinking",
        ("<think>", "</think>"),
    ),
    (
        "obs.use_past_error_logs",
        ("Hey, this is an error in the past",),
    ),
    (
        "obs.use_action_history",
        ("<action>", "click('41')", "click('42')"),
    ),
    (
        "use_memory",
        ("<memory>", "</memory>", "memory A", "memory B"),
    ),
    # (
    #     "obs.use_diff",
    #     ("diff:", "- Step 2", "Identical"),
    # ),
    (
        "use_concrete_example",
        ("# Concrete Example", "<action>\nclick('a324')"),
    ),
    (
        "use_abstract_example",
        ("# Abstract Example",),
    ),
    # (
    #     "action.action_set.multiaction",
    #     ("One or several actions, separated by new lines",),
    # ),
]


def test_shrinking_observation():
    flags = deepcopy(FLAGS_GPT_3_5)
    flags.obs.use_html = True

    prompt_maker = MainPrompt(
        action_set=bgym.HighLevelActionSet(),
        obs_history=OBS_HISTORY,
        actions=ACTIONS,
        memories=MEMORIES,
        thoughts=THOUGHTS,
        previous_plan="1- think\n2- do it",
        step=2,
        flags=flags,
    )

    prompt = str(prompt_maker.prompt)
    new_prompt = str(
        dp.fit_tokens(prompt_maker, max_prompt_tokens=count_tokens(prompt) - 1, max_iterations=7)
    )
    assert count_tokens(new_prompt) < count_tokens(prompt)
    assert "[1] Click me" in prompt
    assert "[1] Click me" in new_prompt
    assert "</html>" in prompt
    assert "</html>" not in new_prompt


@pytest.mark.parametrize("flag_name, expected_prompts", FLAG_EXPECTED_PROMPT)
def test_main_prompt_elements_gone_one_at_a_time(flag_name: str, expected_prompts):

    if flag_name in ["use_thinking", "obs.use_action_history"]:
        return  # TODO design new tests for those two flags

    # Disable the flag
    flags = deepcopy(ALL_TRUE_FLAGS)
    if "." in flag_name:
        prefix, flag_name = flag_name.split(".")
        sub_flags = getattr(flags, prefix)
        setattr(sub_flags, flag_name, False)
    else:
        setattr(flags, flag_name, False)

    if flag_name == "use_memory":
        memories = None
    else:
        memories = MEMORIES

    # Initialize MainPrompt
    prompt = str(
        MainPrompt(
            action_set=flags.action.action_set.make_action_set(),
            obs_history=OBS_HISTORY,
            actions=ACTIONS,
            memories=memories,
            thoughts=THOUGHTS,
            previous_plan="1- think\n2- do it",
            step=2,
            flags=flags,
        ).prompt
    )

    # Verify all elements are not present
    for expected in expected_prompts:
        assert expected not in prompt


def test_main_prompt_elements_present():
    # Make sure the flag is enabled

    # Initialize MainPrompt
    prompt = str(
        MainPrompt(
            action_set=bgym.HighLevelActionSet(),
            obs_history=OBS_HISTORY,
            actions=ACTIONS,
            memories=MEMORIES,
            thoughts=THOUGHTS,
            previous_plan="1- think\n2- do it",
            step=2,
            flags=ALL_TRUE_FLAGS,
        ).prompt
    )
    # Verify all elements are not present
    for _, expected_prompts in FLAG_EXPECTED_PROMPT:
        for expected in expected_prompts:
            assert expected in prompt


if __name__ == "__main__":
    # for debugging
    test_shrinking_observation()
    test_main_prompt_elements_present()
    # for flag, expected_prompts in FLAG_EXPECTED_PROMPT:
    #     test_main_prompt_elements_gone_one_at_a_time(flag, expected_prompts)

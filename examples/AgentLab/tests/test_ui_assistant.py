from agentlab.ui_assistant import make_exp_args
from agentlab.agents.generic_agent import AGENT_4o


def test_make_exp_args():
    """Basic unit test to detect refactoring errors."""
    exp_args = make_exp_args(AGENT_4o)

    assert exp_args.agent_args.flags.action.demo_mode == "default"

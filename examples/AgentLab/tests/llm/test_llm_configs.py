from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
from agentlab.llm.chat_api import BaseModelArgs


def test_llm_configs():

    for _, args in CHAT_MODEL_ARGS_DICT.items():
        assert isinstance(args, BaseModelArgs)

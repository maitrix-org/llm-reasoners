import logging
import tempfile

import pytest
from browsergym.experiments.loop import EnvArgs, ExpArgs

from agentlab.agents.visualwebarena.agent import VisualWebArenaAgentArgs
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT


@pytest.mark.pricy
def test_agent():
    with tempfile.TemporaryDirectory() as exp_dir:
        env_args = EnvArgs(
            task_name="miniwob.click-button",
            task_seed=0,
            max_steps=10,
            headless=True,
        )

        chat_model_args = CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"]

        exp_args = [
            ExpArgs(
                agent_args=VisualWebArenaAgentArgs(
                    temperature=0.1,
                    chat_model_args=chat_model_args,
                ),
                env_args=env_args,
                logging_level=logging.INFO,
            ),
            ExpArgs(
                agent_args=VisualWebArenaAgentArgs(
                    temperature=0.0,
                    chat_model_args=chat_model_args,
                ),
                env_args=env_args,
                logging_level=logging.INFO,
            ),
        ]

        for exp_arg in exp_args:
            exp_arg.agent_args.prepare()
            exp_arg.prepare(exp_dir)

        for exp_arg in exp_args:
            exp_arg.run()
            exp_arg.agent_args.close()

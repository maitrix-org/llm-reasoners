import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from browsergym.experiments.loop import EnvArgs, ExpArgs
from openai import OpenAIError

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import launch_exp
from agentlab.llm.chat_api import BaseModelArgs, CheatMiniWoBLLMArgs
from agentlab.llm.llm_utils import Discussion


def test_generic_agent():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs(),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )

        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 1,
            "cum_reward": 1.0,
            "terminated": True,
            "truncated": False,
            "err_msg": None,
            "stack_trace": None,
            "agent.flags.obs.use_ax_tree": True,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


@dataclass
class CheatMiniWoBLLM_ParseRetry:
    """For unit-testing purposes only. It only work with miniwob.click-test task."""

    n_retry: int
    retry_count: int = 0

    def __call__(self, messages) -> str:
        if self.retry_count < self.n_retry:
            self.retry_count += 1
            return dict(role="assistant", content="I'm retrying")

        if isinstance(messages, Discussion):
            prompt = messages.to_string()
        else:
            prompt = messages[1].get("content", "")
        match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

        if match:
            bid = match.group(1)
            action = f'click("{bid}")'
        else:
            raise Exception("Can't find the button's bid")

        answer = f"""I'm clicking the button as requested.
<action>
{action}
</action>
"""
        return dict(role="assistant", content=answer)

    def get_stats(self):
        return {}


@dataclass
class CheatMiniWoBLLMArgs_ParseRetry(BaseModelArgs):
    n_retry: int = 2
    model_name: str = "test/cheat_miniwob_click_test_parse_retry"

    def make_model(self):
        return CheatMiniWoBLLM_ParseRetry(n_retry=self.n_retry)


@dataclass
class CheatLLM_LLMError:
    """For unit-testing purposes only. Fails to call LLM"""

    n_retry: int = 0
    success: bool = False

    def __call__(self, messages) -> str:
        if self.success:
            if isinstance(messages, Discussion):
                prompt = messages.to_string()
            else:
                prompt = messages[1].get("content", "")
            match = re.search(r"^\s*\[(\d+)\].*button", prompt, re.MULTILINE | re.IGNORECASE)

            if match:
                bid = match.group(1)
                action = f'click("{bid}")'
            else:
                raise Exception("Can't find the button's bid")

            answer = f"""I'm clicking the button as requested.
    <action>
    {action}
    </action>
    """
            return dict(role="assistant", content=answer)
        raise OpenAIError("LLM failed to respond")

    def get_stats(self):
        return {"n_llm_retry": self.n_retry, "n_llm_busted_retry": int(not self.success)}


@dataclass
class CheatLLMArgs_LLMError(BaseModelArgs):
    n_retry: int = 2
    success: bool = False
    model_name: str = "test/cheat_miniwob_click_test_parse_retry"

    def make_model(self):
        return CheatLLM_LLMError(
            n_retry=self.n_retry,
            success=self.success,
        )


def test_generic_agent_parse_retry():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs_ParseRetry(n_retry=2),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # TODO why these tests don't work with ray backend?
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)
        print(result_record)
        target = {
            "stats.cum_n_retry": 2,
            "stats.cum_busted_retry": 0,
            "n_steps": 1,
            "cum_reward": 1.0,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_bust_parse_retry():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatMiniWoBLLMArgs_ParseRetry(n_retry=10),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "stats.cum_n_retry": 5,
            "stats.cum_busted_retry": 1,
            "n_steps": 0,
            "cum_reward": 0,
            "err_msg": None,  # parsing error is considered an agent failure, not a code error
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_llm_error_success():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatLLMArgs_LLMError(n_retry=3, success=True),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "stats.cum_n_llm_retry": 3,
            "n_steps": 1,
            "cum_reward": 1.0,
            "err_msg": None,
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


def test_llm_error_no_success():
    exp_args = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=CheatLLMArgs_LLMError(n_retry=5, success=False),
            flags=FLAGS_GPT_3_5,
        ),
        env_args=EnvArgs(task_name="miniwob.click-test", task_seed=42),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        launch_exp.run_experiments(
            1, [exp_args], Path(tmp_dir) / "generic_agent_test", parallel_backend="joblib"
        )
        result_record = inspect_results.load_result_df(tmp_dir, progress_fn=None)

        target = {
            "n_steps": 0,
            "cum_reward": 0,
            "err_msg": "Exception uncaught by agent or environment in task miniwob.click-test.\nOpenAIError:\nLLM failed to respond",
        }

        for key, target_val in target.items():
            assert key in result_record
            assert result_record[key].iloc[0] == target_val


if __name__ == "__main__":
    # test_generic_agent()
    test_generic_agent_parse_retry()

import math
import tempfile
from pathlib import Path

import pytest
from browsergym.experiments.loop import EnvArgs, ExpArgs

from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5, AGENT_4o_MINI
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments.launch_exp import find_incomplete, run_experiments, non_dummy_count
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import CheatMiniWoBLLMArgs


def test_relaunch_study():
    study_dir = Path(__file__).parent.parent / "data" / "test_study"
    exp_args_list = find_incomplete(study_dir, include_errors=False)

    assert non_dummy_count(exp_args_list) == 1
    assert exp_args_list[0].env_args.task_name == "miniwob.ascending-numbers"

    exp_args_list = find_incomplete(study_dir, include_errors=True)

    assert non_dummy_count(exp_args_list) == 2


def _test_launch_system(backend="ray", cause_timeout=False):

    if cause_timeout:
        wait_time = 10
        avg_step_timeout = 0.5
    else:
        wait_time = 0
        avg_step_timeout = 10

    exp_args_list = []
    for seed in range(3):
        exp_args_list.append(
            ExpArgs(
                agent_args=GenericAgentArgs(
                    chat_model_args=CheatMiniWoBLLMArgs(wait_time=wait_time),
                    flags=FLAGS_GPT_3_5,
                ),
                env_args=EnvArgs(task_name="miniwob.click-test", task_seed=seed, max_steps=5),
            )
        )

    with tempfile.TemporaryDirectory() as tmp_dir:

        study_dir = Path(tmp_dir) / "generic_agent_test"
        run_experiments(
            n_jobs=2,
            exp_args_list=exp_args_list,
            study_dir=study_dir,
            parallel_backend=backend,
            avg_step_timeout=avg_step_timeout,
        )

        results_df = inspect_results.load_result_df(study_dir, progress_fn=None)
        assert len(results_df) == len(exp_args_list)

        for _, row in results_df.iterrows():
            if row.stack_trace is not None:
                print(row.stack_trace)
            if cause_timeout:
                # assert row.err_msg is not None
                assert math.isnan(row.cum_reward) or row.cum_reward == 0
            else:
                assert row.err_msg is None
                assert row.cum_reward == 1.0

        study_summary = inspect_results.summarize_study(results_df)
        assert len(study_summary) == 1
        assert study_summary.std_err.iloc[0] == 0

        if not cause_timeout:
            assert study_summary.n_completed.iloc[0] == "3/3"
            assert study_summary.avg_reward.iloc[0] == 1.0


def test_launch_system_joblib():
    _test_launch_system(backend="joblib")


def test_launch_system_sequntial():
    _test_launch_system(backend="sequential")


def test_launch_system_ray():
    _test_launch_system(backend="ray")


def test_timeout_ray():
    _test_launch_system(backend="ray", cause_timeout=True)


@pytest.mark.pricy
def test_4o_mini_on_miniwob_tiny_test():
    """Run with `pytest -m pricy`."""
    with tempfile.TemporaryDirectory() as tmp_dir:

        study = Study(agent_args=[AGENT_4o_MINI], benchmark="miniwob_tiny_test", dir=tmp_dir)

        study.run(n_jobs=4)

        results_df = inspect_results.load_result_df(study.dir, progress_fn=None)

        for row in results_df.iterrows():
            if row[1].err_msg:
                print(row[1].err_msg)
                print(row[1].stack_trace)

        assert len(results_df) == len(study.exp_args_list)
        summary = inspect_results.summarize_study(results_df)
        print(summary)
        assert len(summary) == 1
        reward = summary.avg_reward.iloc[0]
        assert reward == 1.0


if __name__ == "__main__":
    test_timeout_ray()
    # test_4o_mini_on_miniwob_tiny_test()
    # test_launch_system_ray()
    # test_launch_system_sequntial()

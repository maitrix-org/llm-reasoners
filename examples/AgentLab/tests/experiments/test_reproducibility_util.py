import json
import tempfile
import time
from pathlib import Path

import bgym
import pytest

from agentlab.agents.generic_agent import AGENT_4o_MINI
from agentlab.analyze import inspect_results
from agentlab.experiments import reproducibility_util


@pytest.mark.parametrize(
    "benchmark_name",
    ["miniwob", "workarena_l1", "webarena", "visualwebarena"],
)
def test_get_reproducibility_info(benchmark_name):

    benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name]()

    info = reproducibility_util.get_reproducibility_info(
        "test_agent", benchmark, "test_id", ignore_changes=True
    )

    print("reproducibility info:")
    print(json.dumps(info, indent=4))

    # assert keys in info
    assert "git_user" in info
    assert "benchmark" in info
    assert "benchmark_version" in info
    assert "agentlab_version" in info
    assert "agentlab_git_hash" in info
    assert "agentlab__local_modifications" in info
    assert "browsergym_version" in info
    assert "browsergym_git_hash" in info
    assert "browsergym__local_modifications" in info


# def test_save_reproducibility_info():
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmp_dir = Path(tmp_dir)

#         info1 = reproducibility_util.save_reproducibility_info(
#             study_dir=tmp_dir,
#             info=reproducibility_util.get_reproducibility_info(
#                 agents_args="GenericAgent",
#                 benchmark_name="miniwob",
#                 ignore_changes=True,
#             ),
#         )
#         time.sleep(1)  # make sure the date changes by at least 1s

#         # this should overwrite the previous info since they are the same beside
#         # the date
#         info2 = reproducibility_util.save_reproducibility_info(
#             study_dir=tmp_dir,
#             info=reproducibility_util.get_reproducibility_info(
#                 agents_args="GenericAgent",
#                 benchmark_name="miniwob",
#                 ignore_changes=True,
#             ),
#         )

#         reproducibility_util.assert_compatible(info1, info2)

#         # this should not overwrite info2 as the agent name is different, it
#         # should raise an error
#         with pytest.raises(ValueError):
#             reproducibility_util.save_reproducibility_info(
#                 study_dir=tmp_dir,
#                 info=reproducibility_util.get_reproducibility_info(
#                     agents_args="GenericAgent_alt",
#                     benchmark_name="miniwob",
#                     ignore_changes=True,
#                 ),
#             )

#         # load json
#         info3 = reproducibility_util.load_reproducibility_info(tmp_dir)

#         assert info2 == info3
#         assert info1 != info3

#         test_study_dir = Path(__file__).parent.parent / "data" / "test_study"
#         result_df = inspect_results.load_result_df(test_study_dir, progress_fn=None)
#         report_df = inspect_results.summarize_study(result_df)

#         with pytest.raises(ValueError):
#             reproducibility_util.append_to_journal(
#                 info3, report_df, journal_path=tmp_dir / "journal.csv"
#             )

#         reproducibility_util.append_to_journal(
#             info3, report_df, journal_path=tmp_dir / "journal.csv", strict_reproducibility=False
#         )

#         print((tmp_dir / "journal.csv").read_text())


if __name__ == "__main__":
    # test_set_temp()
    test_get_reproducibility_info("miniwob")
    # test_save_reproducibility_info()

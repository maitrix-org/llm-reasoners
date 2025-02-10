"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging

from agentlab.agents.generic_agent import (
    AGENT_LLAMA3_70B,
    AGENT_LLAMA31_70B,
    RANDOM_SEARCH_AGENT,
    AGENT_4o,
    AGENT_4o_MINI,
)
from agentlab.agents.plan_agent import PLAN_AGENT_4o_MINI, BASELINE_AGENT_4o_MINI
from agentlab.experiments.study import Study


ignore_dependencies = True
avg_step_timeout = 3600

logging.getLogger().setLevel(logging.DEBUG)
save_logs = False
if save_logs:
    logging.basicConfig(filename="agentlab.log", level=logging.DEBUG)

# choose your agent or provide a new agent
agent_args = [PLAN_AGENT_4o_MINI]
# agent_args = [BASELINE_AGENT_4o_MINI]

# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
# benchmark = "miniwob"
# benchmark = "workarena_l1"
# benchmark = "workarena_l2"
# benchmark = "workarena_l3"
benchmark = "webarena"

# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False

# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False

## Number of parallel jobs
n_jobs = 5  # Make sure to use 1 job when debugging in VSCode
# n_jobs = -1  # to use all available cores

tiny_test_task_names = [
    # reddit
    "webarena.27",
    "webarena.66",
    "webarena.399",
    "webarena.409",
    "webarena.552",
    "webarena.580",
    "webarena.596",
    "webarena.600",
    "webarena.610",
    "webarena.615",
    "webarena.620",
    "webarena.627",
    "webarena.630",
    "webarena.635",
    "webarena.640",
    "webarena.650",
    "webarena.671",
    "webarena.718",
    "webarena.720",
    "webarena.726",
    "webarena.731",
    # shopping
    "webarena.21",
    "webarena.50",
    "webarena.118",
    "webarena.147",
    "webarena.158",
    "webarena.165",
    "webarena.189",
    "webarena.225",
    "webarena.239",
    "webarena.260",
    "webarena.269",
    "webarena.277",
    "webarena.280",
    "webarena.281",
    "webarena.282",
    "webarena.283",
    "webarena.284",
    "webarena.298",
    "webarena.313",
    "webarena.319",
    "webarena.323",
    "webarena.330",
    "webarena.333",
    "webarena.351",
    "webarena.432",
    "webarena.465",
    "webarena.520",
    "webarena.531",
    "webarena.575",
    # "webarena.589",
    # "webarena.655",
    # "webarena.691",
    # "webarena.797",
    # gitlab
    # "webarena.45",
    # "webarena.102",
    # "webarena.132",
    # "webarena.156",
    # "webarena.169",
    # "webarena.175",
    # "webarena.207",
    # "webarena.293",
    # "webarena.303",
    # "webarena.343",
    # "webarena.350",
    # "webarena.357",
    # "webarena.389",
    # "webarena.412",
    # "webarena.415",
    # "webarena.418",
    # "webarena.441",
    # "webarena.442",
    # "webarena.446",
    # "webarena.448",
    # "webarena.481",
    # "webarena.523",
    # "webarena.537",
]

if __name__ == "__main__":  # necessary for dask backend
    # import ray
    # ray.init()

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(
            agent_args,
            benchmark,
            logging_level_stdout=logging.WARNING,
            tiny_test_task_names=tiny_test_task_names,
            ignore_dependencies=ignore_dependencies,
            avg_step_timeout=avg_step_timeout,
        )

    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        strict_reproducibility=reproducibility_mode,
        n_relaunch=3,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)

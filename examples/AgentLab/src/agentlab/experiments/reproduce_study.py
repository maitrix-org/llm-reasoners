"""
This script will leverage an old study to reproduce it on the same tasks and
same seeds. Instead of calling the LLM it will reuse the responses from the old
llm. Load the study in agent-xray and look at the Agent Info HTML to compare
the diff in HTML format.
"""

from agentlab.agents.generic_agent.reproducibility_agent import reproduce_study
from agentlab.experiments.exp_utils import RESULTS_DIR


if __name__ == "__main__":

    # replace by your study name
    old_study = "2024-06-03_12-28-51_final_run_miniwob_llama3-70b"

    study = reproduce_study(RESULTS_DIR / old_study)
    n_jobs = 1

    study.run(n_jobs=n_jobs, parallel_backend="joblib", strict_reproducibility=False)

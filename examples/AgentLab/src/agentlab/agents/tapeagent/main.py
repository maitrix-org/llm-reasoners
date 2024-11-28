from agentlab.agents.tapeagent.tapeagent import TapeAgentArgs
from agentlab.experiments import study_generators
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT


def main(benchmark: str, n_jobs: int, reproducibility: bool):
    agent_args = TapeAgentArgs(
        chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"]
    )
    if reproducibility:
        agent_args.set_reproducibility_mode()
    study = study_generators.run_agents_on_benchmark(agent_args, benchmark)
    study.run(n_jobs=n_jobs, parallel_backend="joblib", strict_reproducibility=reproducibility)
    study.append_to_journal(strict_reproducibility=reproducibility)


if __name__ == "__main__":  # necessary for dask backend
    n_jobs = 8  # 1 when debugging in VSCode, -1 to use all available cores
    benchmark = "workarena.l1"
    main(benchmark, n_jobs, reproducibility=True)

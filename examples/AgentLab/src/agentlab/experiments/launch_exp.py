import logging
from importlib import import_module
from pathlib import Path

import bgym
from browsergym.experiments.loop import ExpArgs, yield_all_exp_results

from agentlab.experiments.exp_utils import run_exp


def run_experiments(
    n_jobs,
    exp_args_list: list[ExpArgs],
    study_dir,
    parallel_backend="ray",
    avg_step_timeout=60,
):
    """Run a list of ExpArgs in parallel.

    To ensure optimal parallelism, make sure ExpArgs.depend_on is set correctly
    and the backend is set to dask.

    Args:
        n_jobs: int
            Number of parallel jobs.
        exp_args_list: list[ExpArgs]
            List of ExpArgs objects.
        study_dir: Path
            Directory where the experiments will be saved.
        parallel_backend: str
            Parallel backend to use. Either "joblib", "ray" or "sequential".
            The only backend that supports webarena graph dependencies correctly is ray or sequential.
        avg_step_timeout: int
            Will raise a TimeoutError if the episode is not finished after env_args.max_steps * avg_step_timeout seconds.

    Raises:
        ValueError: If the parallel_backend is not recognized.
    """

    if len(exp_args_list) == 0:
        logging.warning("No experiments to run.")
        return

    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    # if n_jobs == 1 and parallel_backend != "sequential":
    #     logging.warning("Only 1 job, switching to sequential backend.")
    #     parallel_backend = "sequential"

    logging.info(f"Saving experiments to {study_dir}")
    for exp_args in exp_args_list:
        exp_args.agent_args.prepare()
        exp_args.prepare(exp_root=study_dir)
    try:
        if parallel_backend == "joblib":
            from joblib import Parallel, delayed

            # split sequential (should be no longer needed with dependencies)
            sequential_exp_args, exp_args_list = _split_sequential_exp(exp_args_list)

            logging.info(
                f"Running {len(sequential_exp_args)} in sequential first. The remaining {len(exp_args_list)} will be run in parallel."
            )
            for exp_args in sequential_exp_args:
                run_exp(exp_args, avg_step_timeout=avg_step_timeout)

            Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(run_exp)(exp_args, avg_step_timeout=avg_step_timeout)
                for exp_args in exp_args_list
            )

        # dask will be deprecated, as there was issues. use ray instead
        # elif parallel_backend == "dask":
        #     from agentlab.experiments.graph_execution_dask import (
        #         execute_task_graph,
        #         make_dask_client,
        #     )

        #     with make_dask_client(n_worker=n_jobs):
        #         execute_task_graph(exp_args_list)
        elif parallel_backend == "ray":
            from agentlab.experiments.graph_execution_ray import execute_task_graph, ray

            ray.init(num_cpus=n_jobs)
            try:
                execute_task_graph(exp_args_list, avg_step_timeout=avg_step_timeout)
            finally:
                ray.shutdown()
        elif parallel_backend == "sequential":
            for exp_args in exp_args_list:
                run_exp(exp_args, avg_step_timeout=avg_step_timeout)
        else:
            raise ValueError(f"Unknown parallel_backend: {parallel_backend}")
    finally:
        # will close servers even if there is an exception or ctrl+c
        # servers won't be closed if the script is killed with kill -9 or segfaults.
        logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
        for exp_args in exp_args_list:
            exp_args.agent_args.close()
        logging.info("Experiment finished.")


def find_incomplete(study_dir: str | Path, include_errors=True):
    """Find all incomplete experiments for relaunching.

    Note: completed experiments are kept but are replaced by dummy exp_args
    with nothing to run. This help keeping the dependencies between tasks.

    Args:
        study_dir: Path
            The directory where the experiments are saved.
        include_errors: str
            Find all incomplete experiments and relaunch them.
            - "incomplete_only": relaunch only the incomplete experiments.
            - "incomplete_or_error": relaunch incomplete or errors.

    Returns:
        list[ExpArgs]
            List of ExpArgs objects to relaunch.

    Raises:
        ValueError: If the study_dir does not exist.
    """
    study_dir = Path(study_dir)

    if not study_dir.exists():
        raise ValueError(
            f"You asked to relaunch an existing experiment but {study_dir} does not exist."
        )

    exp_result_list = list(yield_all_exp_results(study_dir, progress_fn=None))
    exp_args_list = [_hide_completed(exp_result, include_errors) for exp_result in exp_result_list]
    # sort according to exp_args.order
    exp_args_list.sort(key=lambda exp_args: exp_args.order if exp_args.order is not None else 0)

    job_count = non_dummy_count(exp_args_list)

    if job_count == 0:
        logging.info(f"No incomplete experiments found in {study_dir}.")
        return exp_args_list
    else:
        logging.info(f"Found {job_count} incomplete experiments in {study_dir}.")

    message = f"Make sure the processes that were running are all stopped. Otherwise, "
    f"there will be concurrent writing in the same directories.\n"

    logging.info(message)

    return exp_args_list


def non_dummy_count(exp_args_list: list[ExpArgs]) -> int:
    return sum([not exp_args.is_dummy for exp_args in exp_args_list])


def noop(*args, **kwargs):
    pass


def _hide_completed(exp_result: bgym.ExpResult, include_errors: bool = True):
    """Hide completed experiments from the list.

    This little hack, allows an elegant way to keep the task dependencies for e.g. webarena
    while skipping the tasks that are completed when relaunching.

    Args:
        exp_result: bgym.ExpResult
            The experiment result to hide.
        include_errors: bool
            If True, include experiments that errored.

    Returns:
        ExpArgs
            The ExpArgs object hidden if the experiment is completed.
    """

    hide = False
    if exp_result.status == "done":
        hide = True
    if exp_result.status == "error" and (not include_errors):
        hide = True

    exp_args = exp_result.exp_args
    exp_args.is_dummy = hide  # just to keep track
    exp_args.status = exp_result.status
    if hide:
        # make those function do nothing since they are finished.
        exp_args.run = noop
        exp_args.prepare = noop

    return exp_args


# TODO remove this function once ray backend is stable
def _split_sequential_exp(exp_args_list: list[ExpArgs]) -> tuple[list[ExpArgs], list[ExpArgs]]:
    """split exp_args that are flagged as sequential from those that are not"""
    sequential_exp_args = []
    parallel_exp_args = []
    for exp_args in exp_args_list:
        if getattr(exp_args, "sequential", False):
            sequential_exp_args.append(exp_args)
        else:
            parallel_exp_args.append(exp_args)

    return sequential_exp_args, parallel_exp_args


def _split_path(path: str):
    """Split a path into a module name and an object name."""
    if "/" in path:
        path = path.replace("/", ".")
    module_name, obj_name = path.rsplit(".", 1)
    return module_name, obj_name


def import_object(path: str):
    module_name, obj_name = _split_path(path)
    try:
        module = import_module(module_name)
        obj = getattr(module, obj_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Error importing {path}: {e}")
    return obj

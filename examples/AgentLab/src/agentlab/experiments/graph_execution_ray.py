# import os

# # Disable Ray log deduplication
# os.environ["RAY_DEDUP_LOGS"] = "0"
import logging
import time

import bgym
import ray
from ray.util import state

from agentlab.experiments.exp_utils import _episode_timeout, run_exp

logger = logging.getLogger(__name__)

run_exp = ray.remote(run_exp)


def execute_task_graph(exp_args_list: list[bgym.ExpArgs], avg_step_timeout=60):
    """Execute a task graph in parallel while respecting dependencies using Ray."""

    exp_args_map = {exp_args.exp_id: exp_args for exp_args in exp_args_list}
    task_map = {}

    def get_task(exp_arg: bgym.ExpArgs):
        if exp_arg.exp_id not in task_map:
            # Get all dependency tasks first
            dependency_tasks = [get_task(exp_args_map[dep_key]) for dep_key in exp_arg.depends_on]

            # Create new task that depends on the dependency results
            task_map[exp_arg.exp_id] = run_exp.options(name=f"{exp_arg.exp_name}").remote(
                exp_arg, *dependency_tasks, avg_step_timeout=avg_step_timeout
            )
        return task_map[exp_arg.exp_id]

    # Build task graph
    for exp_arg in exp_args_list:
        get_task(exp_arg)

    max_timeout = max([_episode_timeout(exp_args, avg_step_timeout) for exp_args in exp_args_list])

    return poll_for_timeout(task_map, max_timeout, poll_interval=max_timeout * 0.1)


def poll_for_timeout(tasks: dict[str, ray.ObjectRef], timeout: float, poll_interval: float = 1.0):
    """Cancel tasks that exceeds the timeout

    I tried various different methods for killing a job that hangs. so far it's
    the only one that seems to work reliably (hopefully)

    Args:
        tasks: dict[str, ray.ObjectRef]
            Dictionary of task_id: task_ref
        timeout: float
            Timeout in seconds
        poll_interval: float
            Polling interval in seconds

    Returns:
        dict[str, Any]: Dictionary of task_id: result
    """
    task_list = list(tasks.values())
    task_ids = list(tasks.keys())

    logger.warning(f"Any task exceeding {timeout} seconds will be cancelled.")

    while True:
        ready, not_ready = ray.wait(task_list, num_returns=len(task_list), timeout=poll_interval)
        for task in not_ready:
            elapsed_time = get_elapsed_time(task)
            # print(f"Task {task.task_id().hex()} elapsed time: {elapsed_time}")
            if elapsed_time is not None and elapsed_time > timeout:
                msg = f"Task {task.task_id().hex()} hase been running for {elapsed_time}s, more than the timeout: {timeout}s."
                if elapsed_time < timeout + 60 + poll_interval:
                    logger.warning(msg + " Cancelling task.")
                    ray.cancel(task, force=False, recursive=False)
                else:
                    logger.warning(msg + " Force killing.")
                    ray.cancel(task, force=True, recursive=False)
        if len(ready) == len(task_list):
            results = []
            for task in ready:
                try:
                    result = ray.get(task)
                except Exception as e:
                    result = e
                results.append(result)

            return {task_id: result for task_id, result in zip(task_ids, results)}


def get_elapsed_time(task_ref: ray.ObjectRef):
    task_id = task_ref.task_id().hex()
    task_info = state.get_task(task_id, address="auto")
    if task_info and task_info.start_time_ms is not None:
        start_time_s = task_info.start_time_ms / 1000.0  # Convert ms to s
        current_time_s = time.time()
        elapsed_time = current_time_s - start_time_s
        return elapsed_time
    else:
        return None  # Task has not started yet

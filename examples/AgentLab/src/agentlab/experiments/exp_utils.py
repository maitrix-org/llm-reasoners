import logging
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from time import sleep, time

from browsergym.experiments.loop import ExpArgs, _move_old_exp, yield_all_exp_results
from tqdm import tqdm

logger = logging.getLogger(__name__)  # Get logger based on module name


# TODO move this to a more appropriate place
RESULTS_DIR = os.environ.get("AGENTLAB_EXP_ROOT", None)
if RESULTS_DIR is None:
    RESULTS_DIR = os.environ.get("UI_COPILOT_RESULTS_DIR", None)
if RESULTS_DIR is None:
    logging.info("$AGENTLAB_EXP_ROOT is not defined, Using $HOME/agentlab_results.")
    RESULTS_DIR = Path.home() / "agentlab_results"
else:
    RESULTS_DIR = Path(RESULTS_DIR)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_exp(exp_arg: ExpArgs, *dependencies, avg_step_timeout=60):
    """Run exp_args.run() with a timeout and handle dependencies."""
    # episode_timeout = _episode_timeout(exp_arg, avg_step_timeout=avg_step_timeout)
    # logger.warning(f"Running {exp_arg.exp_id} with timeout of {episode_timeout} seconds.")
    # with timeout_manager(seconds=episode_timeout):
    # this timeout method is not robust enough. using ray.cancel instead
    return exp_arg.run()


def _episode_timeout(exp_arg: ExpArgs, avg_step_timeout=60):
    """Some logic to determine the episode timeout."""
    max_steps = getattr(exp_arg.env_args, "max_steps", None)
    if max_steps is None:
        episode_timeout_global = 10 * 60 * 60  # 10 hours
    else:
        episode_timeout_global = exp_arg.env_args.max_steps * avg_step_timeout

    episode_timeout_exp = getattr(exp_arg, "episode_timeout", episode_timeout_global)

    return min(episode_timeout_global, episode_timeout_exp)


@contextmanager
def timeout_manager(seconds: int = None):
    """Context manager to handle timeouts."""

    if isinstance(seconds, float):
        seconds = max(1, int(seconds))  # make sure seconds is at least 1

    if seconds is None or sys.platform == "win32":
        try:
            logger.warning("Timeouts are not supported on Windows.")
            yield
        finally:
            pass
        return

    def alarm_handler(signum, frame):

        logger.warning(f"Operation timed out after {seconds}s, raising TimeoutError.")
        # send sigint
        # os.kill(os.getpid(), signal.SIGINT) # this doesn't seem to do much I don't know why

        # Still raise TimeoutError for immediate handling
        # This works, but it doesn't seem enough to kill the job
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def add_dependencies(exp_args_list: list[ExpArgs], task_dependencies: dict[str, list[str]] = None):
    """Add dependencies to a list of ExpArgs.

    Args:
        exp_args_list: list[ExpArgs]
            A list of experiments to run.
        task_dependencies: dict
            A dictionary mapping task names to a list of task names that they
            depend on. If None or empty, no dependencies are added.

    Returns:
        list[ExpArgs]
            The modified exp_args_list with dependencies added.

    Raises:
        ValueError: If the task_dependencies are not valid.
    """

    if task_dependencies is None or all([len(dep) == 0 for dep in task_dependencies.values()]):
        # nothing to be done
        return exp_args_list

    for exp_args in exp_args_list:
        exp_args.make_id()  # makes sure there is an exp_id

    exp_args_map = {exp_args.env_args.task_name: exp_args for exp_args in exp_args_list}
    if len(exp_args_map) != len(exp_args_list):
        raise ValueError(
            (
                "Task names are not unique in exp_args_map, "
                "you can't run multiple seeds with task dependencies."
            )
        )

    for task_name in exp_args_map.keys():
        if task_name not in task_dependencies:
            raise ValueError(f"Task {task_name} is missing from task_dependencies")

    # turn dependencies from task names to exp_ids
    for task_name, exp_args in exp_args_map.items():
        exp_args.depends_on = tuple(
            exp_args_map[dep_name].exp_id for dep_name in task_dependencies[task_name]
        )

    return exp_args_list


# Mock implementation of the ExpArgs class with timestamp checks for unit testing
class MockedExpArgs:
    def __init__(self, exp_id, depends_on=None):
        self.exp_id = exp_id
        self.exp_name = f"exp_{exp_id}"
        self.depends_on = depends_on if depends_on else []
        self.start_time = None
        self.end_time = None
        self.env_args = None

    def run(self):
        self.start_time = time()

        # # simulate playright code, (this was causing issues due to python async loop)
        # import playwright.sync_api

        # pw = playwright.sync_api.sync_playwright().start()
        # pw.selectors.set_test_id_attribute("mytestid")
        sleep(3)  # Simulate task execution time
        self.end_time = time()
        return self


def make_seeds(n, offset=42):
    raise DeprecationWarning("This function will be removed. Comment out this error if needed.")
    return [seed + offset for seed in range(n)]


def order(exp_args_list: list[ExpArgs]):
    raise DeprecationWarning("This function will be removed. Comment out this error if needed.")
    """Store the order of the list of experiments to be able to sort them back.

    This is important for progression or ablation studies.
    """
    for i, exp_args in enumerate(exp_args_list):
        exp_args.order = i
    return exp_args_list


# This was an old function for filtering some issue with the experiments.
def hide_some_exp(base_dir, filter: callable, just_test):
    """Move all experiments that match the filter to a new name."""
    raise DeprecationWarning("This function will be removed. Comment out this error if needed.")
    exp_list = list(yield_all_exp_results(base_dir, progress_fn=None))

    msg = f"Searching {len(exp_list)} experiments to move to _* expriments where `filter(exp_args)` is True."
    if just_test:
        msg += f"\nNote: This is a just a test, no experiments will be moved. Set `just_test=False` to move them."

    logging.info(msg)

    exp_list = tqdm(exp_list, desc=f"Filtering experiments.")

    filtered_out = []
    for exp in exp_list:
        if filter(exp):
            if not just_test:
                _move_old_exp(exp.exp_dir)
            filtered_out.append(exp)
    return filtered_out

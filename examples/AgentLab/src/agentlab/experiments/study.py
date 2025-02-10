import gzip
import logging
import pickle
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import bgym
from bgym import Benchmark, EnvArgs, ExpArgs
from slugify import slugify

from agentlab.agents.agent_args import AgentArgs
from agentlab.analyze import inspect_results
from agentlab.experiments import reproducibility_util as repro
from agentlab.experiments.exp_utils import RESULTS_DIR, add_dependencies
from agentlab.experiments.launch_exp import find_incomplete, non_dummy_count, run_experiments

logger = logging.getLogger(__name__)


def make_study(
    agent_args: list[AgentArgs] | AgentArgs,
    benchmark: bgym.Benchmark | str,
    logging_level_stdout=logging.WARNING,
    suffix="",
    comment=None,
    ignore_dependencies=False,
    tiny_test_task_names=None,
):
    """Run a list of agents on a benchmark.

    Args:
        agent_args: list[AgentArgs] | AgentArgs
            The agent configuration(s) to run. *IMPORTANT*: these objects will be pickled and
            unpickled.  Make sure they are imported from a package that is accessible from
            PYTHONPATH. Otherwise, it won't load in agentlab-xray.
        benchmark: bgym.Benchmark | str
            The benchmark to run the agents on. See bgym.DEFAULT_BENCHMARKS for the main ones. You
            can also make your own by modifying an existing one.
        logging_level_stdout: int
            The logging level for the stdout of the main script. Each job will have its own logging
            level that will save into file and can be seen in agentlab-xray.
        suffix: str
            A suffix to add to the study name. This can be useful to keep track of your experiments.
            By default the study name contains agent name, benchmark name and date.
        comment: str
            Extra comments from the authors of this study to be stored in the reproducibility
            information. Leave any extra information that can explain why results could be different
            than expected.
        ignore_dependencies: bool
            If True, ignore the dependencies of the tasks in the benchmark. *Use with caution.* So
            far, only WebArena and VisualWebArena have dependencies between tasks to minimize the
            influence of solving one task before another one. This dependency graph allows
            experiments to run in parallel while respecting task dependencies. However, it still
            can't run more than 4 and, in practice it's speeding up evaluation by a factor of only
            3x compare to sequential executionz. To accelerate execution, you can ignore
            dependencies and run in full parallel. This leads to a decrease in performance of about
            1%-2%, and could be more. Note: ignore_dependencies on VisualWebArena doesn't work.

    Returns:
        Study object or SequentialStudies object if the benchmark requires manual reset after each
        evaluation such as WebArena and VisualWebArena.
    """

    if not isinstance(agent_args, (list, tuple)):
        agent_args = [agent_args]

    if isinstance(benchmark, str):
        benchmark = bgym.DEFAULT_BENCHMARKS[benchmark.lower()]()

    if "webarena" in benchmark.name and len(agent_args) > 1:
        logger.warning(
            "*WebArena* requires manual reset after each evaluation. Running through SequentialStudies."
        )
        studies = []
        for agent in agent_args:
            studies.append(
                Study(
                    [agent],
                    benchmark,
                    logging_level=logging_level_stdout,
                    suffix=suffix,
                    comment=comment,
                    ignore_dependencies=ignore_dependencies,
                )
            )

        return SequentialStudies(studies)
    else:
        return Study(
            agent_args,
            benchmark,
            logging_level=logging_level_stdout,
            suffix=suffix,
            comment=comment,
            ignore_dependencies=ignore_dependencies,
            tiny_test_task_names=tiny_test_task_names,
        )


class AbstractStudy(ABC):
    """Abstract class for a study."""

    dir: Path = None
    suffix: str = ""

    @abstractmethod
    def find_incomplete(self, include_errors=True):
        """Prepare the study for relaunching by finding incomplete experiments"""

    @abstractmethod
    def run(self, n_jobs=1, parallel_backend="ray", strict_reproducibility=False, n_relaunch=3):
        """Run the study"""

    def make_dir(self, exp_root=RESULTS_DIR):
        """Create a directory for the study"""
        if self.dir is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.name}"

            self.dir = Path(exp_root) / dir_name
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, exp_root=RESULTS_DIR):
        """Pickle the study to the directory"""
        # TODO perhaps remove exp_args_list before pickling and when loading bring them from the individual directories

        self.make_dir(exp_root=exp_root)
        with gzip.open(self.dir / "study.pkl.gz", "wb") as f:
            pickle.dump(self, f)

    def get_results(self, suffix="", also_save=True):
        """Recursively load all results from the study directory and summarize them."""
        result_df = inspect_results.load_result_df(self.dir)
        error_report = inspect_results.error_report(result_df, max_stack_trace=3, use_log=True)
        summary_df = inspect_results.summarize_study(result_df)

        if also_save:
            suffix = f"_{suffix}" if suffix else ""
            result_df.to_csv(self.dir / f"result_df{suffix}.csv")
            summary_df.to_csv(self.dir / f"summary_df{suffix}.csv")
            (self.dir / f"error_report{suffix}.md").write_text(error_report)

        return result_df, summary_df, error_report


@dataclass
class Study(AbstractStudy):
    """A study coresponds to one or multiple agents evaluated on a benchmark.

    This is part of the high level API to help keep experiments organized and reproducible.

    Attributes:
        agent_args: list[AgentArgs]
            The agent configuration(s) to run. *IMPORTANT*: these objects will be pickled and
            unpickled.  Make sure they are imported from a package that is accessible from
            PYTHONPATH. Otherwise, it won't load in agentlab-xray.
        benchmark: bgym.Benchmark | str
            The benchmark to run the agents on. See bgym.DEFAULT_BENCHMARKS for the main ones. You
            can also make your own by modifying an existing one.
        dir: Path
            The directory where the study will be saved. If None, a directory will be created in
            RESULTS_DIR.
        suffix: str
            A suffix to add to the study name. This can be useful to keep track of your experiments.
            By default the study name contains agent name, benchmark name and date.
        uuid: str
            A unique identifier for the study.
        reproducibility_info: dict
            Information about the study that may affect the reproducibility of the experiment. e.g.:
            versions of BrowserGym, benchmark, AgentLab...
        logging_level: int
            The logging level for individual jobs.
        logging_level_stdout: int
            The logging level for the stdout of the main script. Each job will have its own logging
            level that will save into file and can be seen in agentlab-xray.
        comment: str
            Extra comments from the authors of this study to be stored in the reproducibility
            information. Leave any extra information that can explain why results could be different
            than expected.
        ignore_dependencies: bool
            If True, ignore the dependencies of the tasks in the benchmark. *Use with caution.* So
            far, only WebArena and VisualWebArena have dependencies between tasks to minimize the
            influence of solving one task before another one. This dependency graph allows
            experiments to run in parallel while respecting task dependencies. However, it still
            can't run more than 4 and, in practice it's speeding up evaluation by a factor of only
            3x compare to sequential executionz. To accelerate execution, you can ignore
            dependencies and run in full parallel. This leads to a decrease in performance of about
            1%-2%, and could be more. Note: ignore_dependencies on VisualWebArena doesn't work.
        avg_step_timeout: int
            The average step timeout in seconds. This is used to stop the experiments if they are
            taking too long. The default is 60 seconds.
        demo_mode: bool
            If True, the experiments will be run in demo mode, which will record videos, and enable
            visual effects for actions.
        tiny_test_task_names: list[str]
            If not None, only run the experiments on the specified tasks.
    """

    agent_args: list[AgentArgs] = None
    benchmark: Benchmark | str = None
    dir: Path = None
    suffix: str = ""  # used for adding a personnal comment to the study name
    uuid: str = None
    reproducibility_info: dict = None
    logging_level: int = logging.DEBUG
    logging_level_stdout: int = logging.WARNING
    comment: str = None  # Extra comments from the authors of this study
    ignore_dependencies: bool = False
    avg_step_timeout: int = 60
    demo_mode: bool = False
    tiny_test_task_names: list[str] = None

    def __post_init__(self):
        """Initialize the study. Set the uuid, and generate the exp_args_list."""
        self.uuid = uuid.uuid4()
        if isinstance(self.benchmark, str):
            self.benchmark = bgym.DEFAULT_BENCHMARKS[self.benchmark.lower()]()
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        self.make_exp_args_list()

        # Test only a subset of tasks if specified
        if self.tiny_test_task_names is not None:
            self.exp_args_list = [
                exp_args
                for exp_args in self.exp_args_list
                if exp_args.env_args.task_name in self.tiny_test_task_names
            ]

    def make_exp_args_list(self):
        """Generate the exp_args_list from the agent_args and the benchmark."""
        self.exp_args_list = _agents_on_benchmark(
            self.agent_args,
            self.benchmark,
            logging_level=self.logging_level,
            logging_level_stdout=self.logging_level_stdout,
            ignore_dependencies=self.ignore_dependencies,
            demo_mode=self.demo_mode,
        )

    def find_incomplete(self, include_errors=True):
        """Find incomplete or errored experiments in the study directory for relaunching.

        Args:
            include_errors: bool
                If True, include errored experiments in the list.

        Returns:
            list[ExpArgs]: The list of all experiments with completed ones replaced by a
                dummy exp_args to keep the task dependencies.
        """
        self.exp_args_list = find_incomplete(self.dir, include_errors=include_errors)
        n_incomplete = non_dummy_count(self.exp_args_list)
        n_error = [
            getattr(exp_args, "status", "incomplete") == "error" for exp_args in self.exp_args_list
        ].count(True)
        return n_incomplete, n_error

    def load_exp_args_list(self):
        logger.info(f"Loading experiments from {self.dir}")
        self.exp_args_list = list(inspect_results.yield_all_exp_results(savedir_base=self.dir))

    def set_reproducibility_info(self, strict_reproducibility=False, comment=None):
        """Gather relevant information that may affect the reproducibility of the experiment

        e.g.: versions of BrowserGym, benchmark, AgentLab...

        Args:
            strict_reproducibility: bool
                If True, all modifications have to be committed before running the experiments.
                Also, if relaunching a study, it will not be possible if the code has changed.
            comment: str
                Extra comment to add to the reproducibility information.
        """
        agent_names = [a.agent_name for a in self.agent_args]
        info = repro.get_reproducibility_info(
            agent_names,
            self.benchmark,
            self.uuid,
            ignore_changes=not strict_reproducibility,
            comment=comment,
        )
        if self.reproducibility_info is not None:
            repro.assert_compatible(
                self.reproducibility_info, info, raise_if_incompatible=strict_reproducibility
            )
        self.reproducibility_info = info

    def run(
        self,
        n_jobs=1,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=3,
        relaunch_errors=True,
    ):

        self.set_reproducibility_info(
            strict_reproducibility=strict_reproducibility, comment=self.comment
        )
        self.save()

        n_exp = len(self.exp_args_list)
        last_error_count = None

        for i in range(n_relaunch):
            logger.info(f"Launching study {self.name} - trial {i + 1} / {n_relaunch}")
            self._run(n_jobs, parallel_backend, strict_reproducibility)

            suffix = f"trial_{i + 1}_of_{n_relaunch}"
            _, summary_df, _ = self.get_results(suffix=suffix)
            logger.info("\n" + str(summary_df))

            n_incomplete, n_error = self.find_incomplete(include_errors=relaunch_errors)

            if n_error / n_exp > 0.3:
                logger.warning(f"More than 30% of the experiments errored. Stopping the study.")
                return

            if last_error_count is not None and n_error >= last_error_count:
                logger.warning(
                    f"Last trial did not reduce the number of errors. Stopping the study."
                )
                return

            if n_incomplete == 0:
                logger.info(f"Study {self.name} finished.")
                return

        logger.warning(
            f"Study {self.name} did not finish after {n_relaunch} trials. There are {n_incomplete} incomplete experiments."
        )

    def _run(self, n_jobs=1, parallel_backend="joblib", strict_reproducibility=False):
        """Run all experiments in the study in parallel when possible.

        Args:
            n_jobs: int
                Number of parallel jobs.
            parallel_backend: str
                Parallel backend to use. Either "joblib", "dask" or "sequential".
            strict_reproducibility: bool
                If True, all modifications have to be committed before running the experiments.
                Also, if relaunching a study, it will not be possible if the code has changed.

        Raises:
            ValueError: If the exp_args_list is None.
        """

        if self.exp_args_list is None:
            raise ValueError("exp_args_list is None. Please set exp_args_list before running.")

        logger.info("Preparing backends...")
        self.benchmark.prepare_backends()
        logger.info("Backends ready.")

        run_experiments(
            n_jobs,
            self.exp_args_list,
            self.dir,
            parallel_backend=parallel_backend,
            avg_step_timeout=self.avg_step_timeout,
        )

    def append_to_journal(self, strict_reproducibility=True):
        """Append the study to the journal.

        Args:
            strict_reproducibility: bool
                If True, incomplete experiments will raise an error.
        """
        _, summary_df, _ = self.get_results()
        repro.append_to_journal(
            self.reproducibility_info,
            summary_df,
            strict_reproducibility=strict_reproducibility,
        )

    @property
    def name(self):
        agent_names = [a.agent_name for a in self.agent_args]
        return _make_study_name(agent_names, [self.benchmark.name], self.suffix)

    def override_max_steps(self, max_steps):
        for exp_args in self.exp_args_list:
            exp_args.env_args.max_steps = max_steps

    @staticmethod
    def load(dir: Path) -> "Study":
        dir = Path(dir)
        study_path = dir / "study.pkl.gz"
        if not study_path.exists() and dir.is_dir():
            # For backward compatibility
            first_result = next(
                inspect_results.yield_all_exp_results(savedir_base=dir, progress_fn=None)
            )
            benchmark_name = first_result.exp_args.env_args.task_name.split(".")[0]
            agent_args = first_result.exp_args.agent_args
            study = Study(agent_args=agent_args, benchmark=benchmark_name, dir=dir)
        else:
            with gzip.open(dir / "study.pkl.gz", "rb") as f:
                study = pickle.load(f)  # type: Study
            study.dir = dir

            # # just a check
            # for i, exp_args in enumerate(study.exp_args_list):
            #     if exp_args.order != i:
            #         logging.warning(f"The order of the experiments is not correct. {exp_args.order} != {i}")

        return study

    @staticmethod
    def load_most_recent(root_dir: Path = None, contains=None) -> "Study":
        return Study.load(get_most_recent_study(root_dir, contains=contains))


def _make_study_name(agent_names, benchmark_names, suffix=None):
    """Make a study name from the agent and benchmark names."""

    # extract unique agent and benchmark names
    agent_names = list(set(agent_names))
    benchmark_names = list(set(benchmark_names))

    if len(agent_names) == 1:
        agent_name = agent_names[0]
    else:
        agent_name = f"{len(agent_names)}_agents"

    if len(benchmark_names) == 1:
        benchmark_name = benchmark_names[0]
    else:
        benchmark_name = f"{len(benchmark_names)}_benchmarks"

    study_name = f"{agent_name}_on_{benchmark_name}_{suffix if suffix else ''}"

    return slugify(study_name, max_length=200, allow_unicode=True)


@dataclass
class SequentialStudies(AbstractStudy):
    """
    Sequential execution of multiple studies.

    This is required for e.g. WebArena, where a server reset is required between evaluations of each agent.
    """

    studies: list[Study]

    @property
    def name(self):
        """The name of the study."""
        agent_names = [a.agent_name for study in self.studies for a in study.agent_args]
        benchmark_names = [study.benchmark.name for study in self.studies]
        return _make_study_name(agent_names, benchmark_names, self.suffix)

    def find_incomplete(self, include_errors=True):
        for study in self.studies:
            study.find_incomplete(include_errors=include_errors)

    def run(self, n_jobs=1, parallel_backend="ray", strict_reproducibility=False, n_relaunch=3):

        # This sequence of of making directories is important to make sure objects are materialized
        # properly before saving. Otherwise relaunch may not work properly.
        self.make_dir()
        for study in self.studies:
            study.make_dir(exp_root=self.dir)

        self.save()

        for study in self.studies:
            study.run(n_jobs, parallel_backend, strict_reproducibility, n_relaunch)
        _, summary_df, _ = self.get_results()
        logger.info("\n" + str(summary_df))
        logger.info(f"SequentialStudies {self.name} finished.")

    def override_max_steps(self, max_steps):
        for study in self.studies:
            study.override_max_steps(max_steps)

    def append_to_journal(self, strict_reproducibility=True):
        for study in self.studies:
            study.append_to_journal(strict_reproducibility=strict_reproducibility)


def get_most_recent_study(
    root_dir: Path = None, date_format: str = "%Y-%m-%d_%H-%M-%S", contains=None
):
    """Return the most recent directory based on the date in the folder name.

    Args:
        root_dir: The directory to search in
        date_format: The format of the date in the folder name
        contains: If not None, only consider folders that contains this string

    Returns:
        Path: The most recent folder satisfying the conditions
    """

    if root_dir is None:
        root_dir = RESULTS_DIR

    most_recent_folder = None
    most_recent_time = datetime.min

    for item in root_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            if contains is not None and contains not in item.name:
                continue
            try:
                folder_date = datetime.strptime("_".join(item.name.split("_")[:2]), date_format)
                if folder_date > most_recent_time:
                    most_recent_time = folder_date
                    most_recent_folder = item
            except (ValueError, IndexError):
                continue

    return most_recent_folder


def set_demo_mode(env_args_list: list[EnvArgs]):
    """Set the demo mode for the experiments. This can be useful for generating videos for demos."""
    for env_args in env_args_list:
        env_args.viewport = {"width": 1280, "height": 720}
        env_args.record_video = True
        env_args.wait_for_user_message = False
        env_args.slow_mo = 1000


def _agents_on_benchmark(
    agents: list[AgentArgs] | AgentArgs,
    benchmark: bgym.Benchmark,
    demo_mode=False,
    logging_level: int = logging.INFO,
    logging_level_stdout: int = logging.INFO,
    ignore_dependencies=False,
):
    """Run one or multiple agents on a benchmark.

    Args:
        agents: list[AgentArgs] | AgentArgs
            The agent configuration(s) to run.
        benchmark: bgym.Benchmark
            The benchmark to run the agents on.
        demo_mode: bool
            If True, the experiments will be run in demo mode.
        logging_level: int
            The logging level for individual jobs.
        logging_level_stdout: int
            The logging level for the stdout.
        ignore_dependencies: bool
            If True, the dependencies will be ignored and all experiments can be run in parallel.

    Returns:
        list[ExpArgs]: The list of experiments to run.

    Raises:
        ValueError: If multiple agents are run on a benchmark that requires manual reset.
    """

    if not isinstance(agents, (list, tuple)):
        agents = [agents]

    if benchmark.name.startswith("visualwebarena") or benchmark.name.startswith("webarena"):
        if len(agents) > 1:
            raise ValueError(
                f"Only one agent can be run on {benchmark.name} since the instance requires manual reset after each evaluation."
            )

    for agent in agents:
        # the agent can adapt (lightly?) to the benchmark
        agent.set_benchmark(benchmark, demo_mode)

    env_args_list = benchmark.env_args_list
    if demo_mode:
        set_demo_mode(env_args_list)

    exp_args_list = []

    for agent in agents:
        for env_args in env_args_list:
            exp_args = ExpArgs(
                agent_args=agent,
                env_args=env_args,
                logging_level=logging_level,
                logging_level_stdout=logging_level_stdout,
            )
            exp_args_list.append(exp_args)

    for i, exp_args in enumerate(exp_args_list):
        exp_args.order = i

    # not required with ray, but keeping around if we would need it for visualwebareana on joblib
    # _flag_sequential_exp(exp_args_list, benchmark)

    if not ignore_dependencies:
        # populate the depends_on field based on the task dependencies in the benchmark
        exp_args_list = add_dependencies(exp_args_list, benchmark.dependency_graph_over_tasks())
    else:
        logger.warning(
            f"Ignoring dependencies for benchmark {benchmark.name}. This could lead to different results."
        )

    return exp_args_list


# def _flag_sequential_exp(exp_args_list: list[ExpArgs], benchmark: Benchmark):
#     if benchmark.name.startswith("visualwebarena"):
#         sequential_subset = benchmark.subset_from_glob("requires_reset", "True")
#         sequential_subset = set(
#             [env_args.task_name for env_args in sequential_subset.env_args_list]
#         )
#         for exp_args in exp_args_list:
#             if exp_args.env_args.task_name in sequential_subset:
#                 exp_args.sequential = True


# def ablation_study(start_agent: AgentArgs, changes, benchmark: str, demo_mode=False):
#     """Ablation study of an agent.

#     Changes is a list of tuples (path_to_attribute, value) to change in the agent
#     configuration.

#     Args:
#         start_agent: AgentArgs
#             The agent configuration to start from.

#         changes: list[tuple]
#             The changes to apply to the agent configuration.

#         benchmark: str
#             The benchmark to use.

#         demo_mode: bool
#             If True, the experiments will be run in demo mode.

#     Returns:
#         Study
#     """
#     agents = args.make_ablation_study(start_agent, changes)
#     study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
#     study.suffix = "ablation_study"
#     return study


# def random_search(
#     random_agent: AgentArgs = RANDOM_SEARCH_AGENT,
#     n_samples=10,
#     benchmark: str = "miniwob",
#     demo_mode=False,
# ):
#     """
#     Random search of AgentArgs (NOTE: not fully tested since refactoring)

#     The random search mechanism will recursively search through dataclasses and
#     dict to find attributes of type args.Choice. It will sample iid and replace
#     with the corresponding value.

#     *WARINING* The standard errror of the experiment will usually be relatively high and
#     the search space is usually big so the false discovery rate will likely be
#     high. Make sure to analyze the results with caution and don't actually draw
#     final conclusions from these experiments.

#     Args:
#         agent: AgentArgs
#             The agent configuration, with some sub-arguments defined as args.Choice.

#         n_samples: int
#             The number of samples to take.

#         benchmark: str
#             The benchmark to use.

#         demo_mode: bool
#             If True, the experiments will be run in demo mode.

#     Returns:
#         Study
#     """
#     agents = args.sample_and_expand_cross_product(random_agent, n_samples)
#     study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
#     study.suffix = "random_search"
#     return study

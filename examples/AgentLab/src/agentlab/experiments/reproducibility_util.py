import csv
import logging
import os
import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path

import bgym
import pandas as pd
from git import InvalidGitRepositoryError, Repo
from git.config import GitConfigParser

import agentlab
from agentlab.experiments.exp_utils import RESULTS_DIR


def _get_repo(module):
    return Repo(Path(module.__file__).resolve().parent, search_parent_directories=True)


def _get_benchmark_version(benchmark: bgym.Benchmark) -> str:
    benchmark_name = benchmark.name

    if hasattr(benchmark, "get_version"):
        return benchmark.get_version()

    # in between 2 pull requests
    if benchmark_name.startswith("miniwob"):
        return metadata.distribution("browsergym.miniwob").version
    elif benchmark_name.startswith("workarena"):
        return metadata.distribution("browsergym.workarena").version
    elif benchmark_name.startswith("webarena"):
        return metadata.distribution("browsergym.webarena").version
    elif benchmark_name.startswith("visualwebarena"):
        return metadata.distribution("browsergym.visualwebarena").version
    elif benchmark_name.startswith("weblinx"):
        try:
            return metadata.distribution("weblinx_browsergym").version
        except metadata.PackageNotFoundError:
            return "0.0.1rc1"
    elif benchmark_name.startswith("assistantbench"):
        return metadata.distribution("browsergym.assistantbench").version
    else:
        raise ValueError(f"Unknown benchmark {benchmark_name}")


def _get_git_username(repo: Repo) -> str:
    """
    Retrieves the first available Git username from various sources.

    Note: overlycomplex designed by Claude and not fully tested.

    This function checks multiple locations for the Git username in the following order:
    1. Repository-specific configuration
    2. GitHub API (if the remote is a GitHub repository)
    3. Global Git configuration
    4. System Git configuration
    5. Environment variables (GIT_AUTHOR_NAME and GIT_COMMITTER_NAME)

    Args:
        repo (Repo): A GitPython Repo object representing the Git repository.

    Returns:
        str: The first non-None username found, or None if no username is found.
    """
    # Repository-specific configuration
    try:
        username = repo.config_reader().get_value("user", "name", None)
        if username:
            return username
    except Exception:
        pass

    try:
        # GitHub username
        remote_url = repo.remotes.origin.url
        if "github.com" in remote_url:
            import json
            import re
            import urllib.request

            match = re.search(r"github\.com[:/](.+)/(.+)\.git", remote_url)
            if match:
                owner, repo_name = match.groups()
                api_url = f"https://api.github.com/repos/{owner}/{repo_name}"
                with urllib.request.urlopen(api_url) as response:
                    data = json.loads(response.read().decode())
                    username = data["owner"]["login"]
                    if username:
                        return username
    except Exception:
        pass

    try:
        # Global configuration
        username = GitConfigParser(repo.git.config("--global", "--list"), read_only=True).get_value(
            "user", "name", None
        )
        if username:
            return username
    except Exception:
        pass

    try:
        # System configuration
        username = GitConfigParser(repo.git.config("--system", "--list"), read_only=True).get_value(
            "user", "name", None
        )
        if username:
            return username
    except Exception:
        pass

    # Environment variables
    return os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("GIT_COMMITTER_NAME")


def _get_git_info(module, changes_white_list=()) -> tuple[str, list[tuple[str, Path]]]:
    """
    Retrieve comprehensive git information for the given module.

    This function attempts to find the git repository containing the specified
    module and returns the current commit hash and a comprehensive list of all
    files that contribute to the repository's state.

    Args:
        module: The Python module object to check for git information.
        changes_white_list: A list of file paths to ignore when checking for changes.

    Returns:
        tuple: A tuple containing two elements:
            - str or None: The current git commit hash, or None if not a git repo.
            - list of tuple: A list of (status, Path) tuples for all modified files.
              Empty list if not a git repo. Status can be 'M' (modified), 'A' (added),
              'D' (deleted), 'R' (renamed), 'C' (copied), 'U' (updated but unmerged),
              or '??' (untracked).
    """

    try:
        repo = _get_repo(module)

        git_hash = repo.head.object.hexsha

        modified_files = []

        # Staged changes
        staged_changes = repo.index.diff(repo.head.commit)
        for change in staged_changes:
            modified_files.append((change.change_type, Path(change.a_path)))

        # Unstaged changes
        unstaged_changes = repo.index.diff(None)
        for change in unstaged_changes:
            modified_files.append((change.change_type, Path(change.a_path)))

        # Untracked files
        untracked_files = repo.untracked_files
        for file in untracked_files:
            modified_files.append(("??", Path(file)))

        # wildcard matching from white list
        modified_files_filtered = []
        for status, file in modified_files:
            if any(file.match(pattern) for pattern in changes_white_list):
                continue
            modified_files_filtered.append((status, file))

        return git_hash, modified_files_filtered
    except InvalidGitRepositoryError:
        return None, []


def get_reproducibility_info(
    agent_names: str | list[str],
    benchmark: bgym.Benchmark,
    study_id: str = "",
    comment=None,
    changes_white_list=(  # Files that are often modified during experiments but do not affect reproducibility
        "*/reproducibility_script.py",
        "*reproducibility_journal.csv",
        "*main.py",
        "*inspect_results.ipynb",
    ),
    ignore_changes=False,
):
    """
    Retrieve a dict of information that could influence the reproducibility of an experiment.
    """
    from browsergym import core

    import agentlab

    if isinstance(agent_names, str):
        agent_names = [agent_names]

    try:
        repo = _get_repo(agentlab)
    except InvalidGitRepositoryError:
        repo = None

    info = {
        "git_user": _get_git_username(repo),
        "agent_names": agent_names,
        "benchmark": benchmark.name,
        "study_id": study_id,
        "comment": comment,
        "benchmark_version": _get_benchmark_version(benchmark),
        "date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "os": f"{platform.system()} ({platform.version()})",
        "python_version": platform.python_version(),
        "playwright_version": metadata.distribution("playwright").version,
    }

    def add_git_info(module_name, module):
        git_hash, modified_files = _get_git_info(module, changes_white_list)

        modified_files_str = "\n".join([f"  {status}: {file}" for status, file in modified_files])

        if len(modified_files) > 0:
            msg = (
                f"Module {module_name} has uncommitted changes. "
                f"Modified files:  \n{modified_files_str}\n"
            )
            if ignore_changes:
                logging.warning(
                    msg + "Ignoring changes as requested and proceeding to experiments."
                )
            else:
                raise ValueError(
                    msg + "Please commit or stash your changes before running the experiment."
                )

        info[f"{module_name}_version"] = module.__version__
        info[f"{module_name}_git_hash"] = git_hash
        info[f"{module_name}__local_modifications"] = modified_files_str

    add_git_info("agentlab", agentlab)
    add_git_info("browsergym", core)
    return info


def assert_compatible(info: dict, old_info: dict, raise_if_incompatible=True):
    """Make sure that the two info dicts are compatible."""
    # TODO may need to adapt if there are multiple agents, and the re-run on
    # error only has a subset of agents. Hence old_info.agent_name != info.agent_name
    for key in info.keys():
        if key in ("date", "avg_reward", "std_err", "n_completed", "n_err"):
            continue
        if info[key] != old_info[key]:
            _raise_or_warn(
                f"Reproducibility info already exist and is not compatible."
                f"Key {key} has changed from {old_info[key]} to {info[key]}."
                f"Set strict_reproducibility=False to bypass this error.",
                raise_error=raise_if_incompatible,
            )


def _raise_or_warn(msg, raise_error=True):
    if raise_error:
        raise ValueError(msg)
    else:
        logging.warning(msg)


def _verify_report(report_df: pd.DataFrame, agent_names=list[str], strict_reproducibility=True):

    report_df = report_df.reset_index()

    unique_agent_names = report_df["agent.agent_name"].unique()
    if set(agent_names) != set(unique_agent_names):
        raise ValueError(
            f"Agent names in the report {unique_agent_names} do not match the agent names {agent_names}."
        )
    if len(set(agent_names)) != len(agent_names):
        raise ValueError(f"Duplicate agent names {agent_names}.")

    report_df = report_df.set_index("agent.agent_name", inplace=False)

    for idx in report_df.index:
        n_err = report_df.loc[idx, "n_err"].item()
        n_completed, n_total = report_df.loc[idx, "n_completed"].split("/")
        if n_err > 0:
            _raise_or_warn(
                f"Experiment {idx} has {n_err} errors. Please rerun the study and make sure all tasks are completed.",
                raise_error=strict_reproducibility,
            )
        if n_completed != n_total:
            _raise_or_warn(
                f"Experiment {idx} has {n_completed} completed tasks out of {n_total}. "
                f"Please rerun the study and make sure all tasks are completed.",
                raise_error=strict_reproducibility,
            )
    return report_df


def _get_csv_headers(file_path: str) -> list[str]:
    with open(file_path, "r", newline="") as file:
        reader = csv.reader(file)
        try:
            headers = next(reader)
        except StopIteration:
            headers = None
    return headers


def _add_result_to_info(info: dict, report_df: pd.DataFrame):
    """Extracts the results from the report and adds them to the info dict inplace"""

    for key in ("avg_reward", "std_err", "n_err", "n_completed"):
        value = report_df.loc[info["agent_name"], key]
        if hasattr(value, "item"):
            value = value.item()
        info[key] = value


def append_to_journal(
    info, report_df: pd.DataFrame, journal_path=None, strict_reproducibility=True
):
    """Append the info and results to the reproducibility journal."""
    if journal_path is None:
        try:
            _get_repo(agentlab)  # if not based on git clone, this will raise an error
            journal_path = (
                Path(agentlab.__file__).parent.parent.parent / "reproducibility_journal.csv"
            )
        except InvalidGitRepositoryError:
            logging.warning(
                "Could not find a git repository. Saving the journal to the results directory."
                "To add to the journal, git clone agentlab and use `pip install -e .`"
            )
            journal_path = RESULTS_DIR / "reproducibility_journal.csv"

    logging.info(f"Appending to journal {journal_path}")

    if len(report_df) != len(info["agent_names"]):
        raise ValueError(
            "Mismatch between the number of agents in reproducibility info and the summary report."
        )

    report_df = _verify_report(
        report_df, info["agent_names"], strict_reproducibility=strict_reproducibility
    )

    rows = []
    headers = None
    if journal_path.exists():
        headers = _get_csv_headers(journal_path)

    if headers is None:  # first creation
        headers = list(info.keys())
        headers[headers.index("agent_names")] = "agent_name"
        rows.append(headers)

    for agent_name in info["agent_names"]:
        info_copy = info.copy()
        del info_copy["agent_names"]
        info_copy["agent_name"] = agent_name

        _add_result_to_info(info_copy, report_df)

        rows.append([str(info_copy[key]) for key in headers])

    with open(journal_path, "a", newline="") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)

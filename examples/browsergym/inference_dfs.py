import argparse
import os
import pickle
import time

from reasoners import Reasoner
from reasoners.algorithm import DFS
from reasoners.lm import OpenAIModel
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import EnvArgs

from gym_env import EnvironmentGym
from search_config import SearchConfigBrowsergym
from utils.misc import obs_preprocessor
from utils.parse import parse_common_arguments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a task with specified parameters."
    )
    parse_common_arguments(parser)

    # DFS parameters
    parser.add_argument(
        "--total_states", type=int, default=100, help="Total states for DFS."
    )
    parser.add_argument(
        "--max_per_state", type=int, default=3, help="Max per state for DFS."
    )
    parser.add_argument(
        "--depth", type=int, default=10, help="Depth limit for DFS."
    )

    return parser.parse_args()


def run_task(args):
    browser_action_set = HighLevelActionSet(
        subsets=[args.action_set],
        strict=False,
        multiaction=True,
        demo_mode="off",
    )

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=args.task_seed,
        max_steps=args.max_steps,
        headless=True,
        record_video=True,
    )

    exp_dir = os.path.join(args.exp_dir, args.task_name)
    os.makedirs(exp_dir, exist_ok=True)

    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,
        exp_dir=exp_dir,
    )

    llm = OpenAIModel(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        backend=args.backend,
    )

    world_model = EnvironmentGym(env=env, obs_preprocessor=obs_preprocessor)
    search_config = SearchConfigBrowsergym(
        action_set=browser_action_set,
        n_proposals=10,
        llm=llm,
        use_axtree=True,
        use_html=False,
        use_screenshot=False,
    )
    algorithm = DFS(
        total_states=args.total_states,
        max_per_state=args.max_per_state,
        depth=args.depth,
    )

    reasoner = Reasoner(world_model, search_config, algorithm)

    plan_result = reasoner()

    with open(f"{exp_dir}/result.pkl", "wb") as f:
        pickle.dump(plan_result, f)

    env.close()

    return plan_result.terminal_state and plan_result.terminal_state.reward == 1.0


if __name__ == "__main__":
    args = parse_arguments()

    start_time = time.time()
    success = run_task(args)

    if success:
        print("Task completed successfully.")
    else:
        print(
            "Task didn't reach the goal. Please check the detailed result w/ visualization (python visualize.py --task_name <task_name>).",
        )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

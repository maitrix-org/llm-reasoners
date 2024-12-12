from reasoners import Reasoner
from reasoners.algorithm import MCTS
from reasoners.lm import OpenAIModel
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import EnvArgs

from gym_env import EnvironmentGym
from searchconfig import SearchConfigBrowsergym
from utils.misc import obs_preprocessor

import os
import pickle
import sys


def run_task(task_name: str, task_seed: int = 42):
    browser_action_set = HighLevelActionSet(
        subsets=["webarena"],
        strict=False,
        multiaction=True,
        demo_mode="off",  # 'default' is on
    )

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=task_seed,
        max_steps=100,
        headless=True,
        record_video=True,
    )

    if not os.path.exists(f"./results/tree-search/{task_name}"):
        os.makedirs(f"./results/tree-search/{task_name}")

    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,
        exp_dir=f"./results/tree-search/{task_name}",
    )

    llm = OpenAIModel(model="gpt-4o-mini")

    world_model = EnvironmentGym(env=env, obs_preprocessor=obs_preprocessor)
    search_config = SearchConfigBrowsergym(
        action_set=browser_action_set, n_proposals=10, llm=llm, use_axtree=True, use_html=False, use_screenshot=False)
    algorithm = MCTS(n_iters=10, depth_limit=10, w_exp=10**.5,
                     uct_with_fast_reward=True, disable_tqdm=False, output_trace_in_each_iter=True)
    reasoner = Reasoner(world_model, search_config, algorithm)

    result_rap = reasoner()

    with open(f"./results/tree-search/{task_name}/result.pkl", "wb") as f:
        pickle.dump(result_rap, f)

    env.close()

    return result_rap.terminal_state and result_rap.terminal_state.reward == 1.0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <task_name>")
        sys.exit(1)

    task_name = sys.argv[1]
    success = run_task(task_name)

    if success:
        print("Task completed successfully.")
    else:
        print("Task failed.")

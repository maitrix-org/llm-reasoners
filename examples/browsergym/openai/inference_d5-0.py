import argparse
import os
import pickle
import time

from reasoners import Reasoner
from reasoners.algorithm import MCTS
from reasoners.lm import OpenAIModel
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import EnvArgs

from gym_env import EnvironmentGym
from search_config import SearchConfigBrowsergym
from utils.misc import obs_preprocessor
from utils.parse import parse_common_arguments

import traceback
import signal 
from utils.timeout import timeout_handler
signal.signal(signal.SIGALRM, timeout_handler)


depth=5
portion = 0
name = f"d{depth}-{portion}"

def run_exp(exp_name: str, task_names: str):
    exp_dir = f"./results/{exp_name}"
    exp_dir_abspath = os.path.abspath(exp_dir)
    if not os.path.exists(exp_dir_abspath):
        os.makedirs(exp_dir)
        with open(f"{exp_dir}/status.txt", "w+") as f:
            f.write("")
    
    status = open(f"{exp_dir}/status.txt", "r+").readlines()
    completed_tasks = [line.strip().split(" ")[0] for line in status]

    for task_name in task_names:
        with open(f"{exp_dir}/status.txt", "a") as f:
            if task_name in completed_tasks:
                print(f"skipping {task_name}")
                continue
            else:
                print(f"working on {task_name}")
                try:
                    signal.alarm(60 * 120) # if it runs for more than two hours, just call it a failure
                    success = run_task(exp_name, task_name)
                    f.write(f"{task_name} {success}\n")
                except Exception as e:
                    f.write(f"{task_name} ERROR\n")
                    f.write(traceback.format_exc())
                finally:
                    signal.alarm(0)


def run_task(exp_name: str, task_name: str) -> bool:

    start = time.time()

    browser_action_set = HighLevelActionSet(
        subsets=["webarena"],
        strict=False,
        multiaction=True,
        demo_mode="off",
    )

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=42,
        max_steps=depth,
        headless=True,
        record_video=True,
    )

    task_dir = os.path.join("./results", exp_name, task_name)
    os.makedirs(task_dir, exist_ok=True)

    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,
        exp_dir=task_dir,
    )

    llm = OpenAIModel(
        model="gpt-4o-mini",
        temperature=0.7,
        task_dir=task_dir
    )

    world_model = EnvironmentGym(env=env, obs_preprocessor=obs_preprocessor, task_dir=task_dir)

    # greedy search
    search_config = SearchConfigBrowsergym(
        action_set=browser_action_set,
        n_proposals=10,
        llm=llm,
        use_axtree=True,
        use_html=False,
        use_screenshot=False,
        task_dir=task_dir
    )
    algorithm = MCTS(
        n_iters=10,
        depth_limit=depth,
        w_exp=10**0.5,
        uct_with_fast_reward=True,
        disable_tqdm=False,
        output_trace_in_each_iter=True,
        task_dir=task_dir
    )

    reasoner = Reasoner(world_model, search_config, algorithm)

    plan_result = reasoner()

    with open(f"{task_dir}/result.pkl", "wb") as f:
        pickle.dump(plan_result, f)

    env.close()

    end = time.time()

    with open(f"{task_dir}/time.txt", "a+") as f:
        f.write(f"total time taken: {end - start}\n")

    return plan_result.terminal_state and plan_result.terminal_state.reward == 1.0

first_half_tasks = [
"webarena.27",
"webarena.28",
"webarena.29",
"webarena.30",
"webarena.31",
"webarena.66",
"webarena.67",
"webarena.68",
"webarena.69",
"webarena.399",
"webarena.400",
"webarena.401",
"webarena.402",
"webarena.403",
"webarena.404",
"webarena.405",
"webarena.406",
"webarena.407",
"webarena.408",
"webarena.409",
"webarena.410",
"webarena.580",
"webarena.581",
"webarena.582",
"webarena.583",
"webarena.584",
"webarena.595",
"webarena.596",
"webarena.597",
"webarena.598",
"webarena.599",
"webarena.600",
"webarena.601",
"webarena.602",
"webarena.603",
"webarena.604",
"webarena.605",
"webarena.606",
"webarena.607",
"webarena.608",
"webarena.609",
"webarena.610",
"webarena.611",
"webarena.612",
"webarena.613",
"webarena.614",
"webarena.615",
"webarena.616",
"webarena.617",
"webarena.618",
"webarena.619",
"webarena.620",
]

second_half_tasks = [
"webarena.621",
"webarena.622",
"webarena.623",
"webarena.624",
"webarena.625",
"webarena.626",
"webarena.627",
"webarena.628",
"webarena.629",
"webarena.630",
"webarena.631",
"webarena.632",
"webarena.633",
"webarena.634",
"webarena.635",
"webarena.636",
"webarena.637",
"webarena.638",
"webarena.639",
"webarena.640",
"webarena.641",
"webarena.642",
"webarena.643",
"webarena.644",
"webarena.645",
"webarena.646",
"webarena.647",
"webarena.648",
"webarena.649",
"webarena.650",
"webarena.651",
"webarena.652",
"webarena.714",
"webarena.715",
"webarena.716",
"webarena.717",
"webarena.718",
"webarena.719",
"webarena.720",
"webarena.721",
"webarena.722",
"webarena.723",
"webarena.724",
"webarena.725",
"webarena.726",
"webarena.727",
"webarena.728",
"webarena.729",
"webarena.730",
"webarena.731",
"webarena.732",
"webarena.733",
"webarena.734",
"webarena.735"
]

run_exp(name, first_half_tasks if portion == 0 else second_half_tasks)

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir")
parser.add_argument("--total-portions")
parser.add_argument("--portion-idx")
parser.add_argument("--tag-skip-count")

parser.add_argument("--model")
parser.add_argument("--backend")
parser.add_argument("--mcts-iterations")
parser.add_argument("--mcts-depth")
parser.add_argument("--n-proposals")
args = parser.parse_args()

from utils.tasks import get_tag_prev_exps_count, filter_out_completed_tasks, get_tasks_subset_for_portion

results_dir = args.results_dir

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

tag=f"d{args.mcts_depth}"
filtered_tasks = filter_out_completed_tasks(results_dir, tag)
tasks_subset = get_tasks_subset_for_portion(int(args.total_portions), int(args.portion_idx), tasks=filtered_tasks)

exp_name=f"d{args.mcts_depth}-{int(args.tag_skip_count) + int(args.portion_idx)}"

def run_exp(exp_name: str, task_names: str):

    exp_dir = f"{results_dir}/{exp_name}"
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
                    signal.alarm(60 * 180) # if it runs for more than three hours, call it a failure
                    # sometimes doesn't go off if the thread is blocked during something like step()
                    success = run_task(exp_name, task_name)
                    f.write(f"{task_name} {success}\n")
                except Exception as e:
                    f.write(f"{task_name} ERROR - {type(e).__name__}\n")
                    f.write(traceback.format_exc())
                finally:
                    signal.alarm(0)
                    print("sleeping for 3 seconds before starting next task")
                    time.sleep(3) # perhaps inducing a cd here will remove the no running event loop errors? 


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
        max_steps=int(args.mcts_depth),
        headless=True,
        record_video=False,
    )

    task_dir = os.path.join(results_dir, exp_name, task_name)
    os.makedirs(task_dir, exist_ok=True)

    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,
        exp_dir=task_dir,
    )

    llm = OpenAIModel(
        backend=args.backend,
        model=args.model,
        # model="DeepSeek-R1-Distill-Qwen-32B",
        # model="Qwen2.5-32B-Instruct",
        task_dir=task_dir
    )

    world_model = EnvironmentGym(env=env, obs_preprocessor=obs_preprocessor, task_dir=task_dir)

    # greedy search
    search_config = SearchConfigBrowsergym(
        action_set=browser_action_set,
        llm=llm,
        n_proposals=int(args.n_proposals), 
        proposal_n_retry=4, evaluation_n_retry=4,
        proposal_temperature=0.6, proposal_max_tokens=8192,
        evaluation_temperature=0.6, evaluation_max_tokens=8192,
        use_axtree=True,
        use_html=False,
        use_screenshot=False,
        task_dir=task_dir
    )
    algorithm = MCTS(
        n_iters=int(args.mcts_iterations),
        depth_limit=int(args.mcts_depth),
        w_exp=10**0.5,
        uct_with_fast_reward=True,
        disable_tqdm=False,
        output_trace_in_each_iter=True,
        task_dir=task_dir
    )

    reasoner = Reasoner(world_model, search_config, algorithm)

    try:
        plan_result = reasoner()

        with open(f"{task_dir}/result.pkl", "wb") as f:
            pickle.dump(plan_result, f)

        env.close()

        end = time.time()

        with open(f"{task_dir}/time.txt", "a+") as f:
            f.write(f"total time taken: {end - start}\n")

        return plan_result.terminal_state and plan_result.terminal_state.reward == 1.0
    except Exception as e:
        env.close()
        raise e

# print(tasks)
print(exp_name)
print(len(tasks_subset))
print(tasks_subset)
run_exp(exp_name, tasks_subset)
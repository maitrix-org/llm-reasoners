import argparse
import os
import pickle
import time

from reasoners import Reasoner
from reasoners.algorithm import MCTS
from reasoners.lm import OpenAIModel
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import EnvArgs

from examples.browsergym.gym_env import EnvironmentGym
from examples.browsergym.searchconfig import SearchConfigBrowsergym
from examples.browsergym.utils.misc import obs_preprocessor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a task with specified parameters."
    )
    # Task parameters
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task to run. e.g., webarena.<task_id>. Note you have to host the task website somewhere.",
    )
    parser.add_argument("--task_seed", type=int, default=42, help="Seed for the task.")
    # Path parameters
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results/tree-search",
        help="Directory to save the results.",
    )
    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use in the demo, while you can adapt to any other model from `reasoners.lm`.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the model."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Maximum tokens for the model."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "sglang"],
        help="Backend for the model. Currently support `openai` and `sglang`.",
    )
    # MCTS parameters
    parser.add_argument(
        "--n_iters", type=int, default=2, help="Number of iterations for MCTS."
    )
    parser.add_argument(
        "--depth_limit", type=int, default=10, help="Depth limit for MCTS."
    )
    parser.add_argument(
        "--w_exp",
        type=float,
        default=10**0.5,
        help="Exploration weight of the UCT score for MCTS.",
    )
    # Environment parameters
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15,
        help="Maximum steps allowed for the environment.",
    )
    parser.add_argument(
        "--action_set", type=str, default="webarena", help="Action set to use."
    )
    parser.add_argument(
        "--use_axtree",
        type=bool,
        default=True,
        help="Use a11y tree for the observation to LLM.",
    )
    parser.add_argument(
        "--use_html",
        type=bool,
        default=False,
        help="Use HTML for the observation to LLM.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=bool,
        default=False,
        help="Use screenshot for the observation to LLM. Make sure the model can process images if set to `True`.",
    )
    parser.add_argument(
        "--record_video",
        type=bool,
        default=False,
        help="Record the video of the browser.",
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
    algorithm = MCTS(
        n_iters=args.n_iters,
        depth_limit=args.depth_limit,
        w_exp=args.w_exp,
        uct_with_fast_reward=True,
        disable_tqdm=False,
        output_trace_in_each_iter=True,
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

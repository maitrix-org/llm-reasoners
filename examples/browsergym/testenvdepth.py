# there seems to be an issue where the experiments cannot go past depth 20 even if the env max_depth is set to 100. the max_depth parameter evidently should work as when testing with max_depth 1, it is controlled properly.

# this file serves as a sanity check to see if setting max depth to 100, then repeating a no_op action over and over would be able to reach the max depth. if not, then you know that something is fucking with you in browsergym. if not, then it's probably an issue w. mcts or the search configruation.

# ok spamming click 42 doesn't seem to trigger the env shutdown. need to go in and look to see how exactly the env max step termination is being controlled.

# ok. i'm stupid. it just flips the truncated flag to true. i completely forgot about this. let me see how i'm handling this in the search config.
# def is_terminal(self, state: StateGym) -> bool:
# return state.terminated or state.truncated or state.step_idx >= self.max_steps

# so yeah. the search algorithm is being controlled by a check to see if it's a terminal node. one of these conditions is tripping at state 20. it's probably not state.truncated or state.terminated, as those as coming directly from the env itself, and this file's test suggests that the env is perfectly fine and working as expected. so it's probably state.step_idx >= self.max_steps?


# FUUUUUUUUUUUUUUUUUUCK. YEP. THAT's exactly what it was. there's another max_steps being considered here. i'm a fucking idiot.
#     def __init__(self, env: gym.Env, env_seed: int = 42, max_steps=20, obs_preprocessor: Optional[Callable[[dict], dict]] = None, task_dir: str = None):
# self.env = env
# self.env_seed = env_seed
# self.obs_preprocessor = obs_preprocessor
# self.max_steps = max_steps
# self.env_current_obs: dict = None
# self.task_dir = task_dir

# why even is this check here. it's redundant with the browserenv max_episode_steps. it's functionality completely overlaps with truncated. i just made a bad assumption that it as doing something unique.

# well this should be the issue found.


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


depth = 100
portion = 0
name = f"test_d{depth}-{portion}"


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
                    # if it runs for more than three hours, call it a failure
                    signal.alarm(60 * 180)
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
        # max_steps=3,
        headless=False,
        record_video=True,
    )

    task_dir = os.path.join("./results", exp_name, task_name)
    os.makedirs(task_dir, exist_ok=True)

    env = env_args.make_env(
        action_mapping=browser_action_set.to_python_code,
        exp_dir=task_dir,
    )

    env.reset()
    print(env)
    for idx in range(depth):
        print(idx)
        obs, reward, terminated, truncated, step_info = env.step(
            "noop(10)\n")
        print(terminated, truncated)

    # llm = OpenAIModel(
    #     model="gpt-4o-mini",
    #     temperature=0.7,
    #     task_dir=task_dir
    # )

    # world_model = EnvironmentGym(
    #     env=env, obs_preprocessor=obs_preprocessor, task_dir=task_dir)

    # # greedy search
    # search_config = SearchConfigBrowsergym(
    #     action_set=browser_action_set,
    #     n_proposals=10,
    #     llm=llm,
    #     use_axtree=True,
    #     use_html=False,
    #     use_screenshot=False,
    #     task_dir=task_dir
    # )
    # algorithm = MCTS(  # no search happening here
    #     n_iters=1,
    #     depth_limit=depth,
    #     w_exp=10**0.5,
    #     uct_with_fast_reward=True,
    #     disable_tqdm=False,
    #     output_trace_in_each_iter=True,
    #     task_dir=task_dir
    # )

    # reasoner = Reasoner(world_model, search_config, algorithm)

    # plan_result = reasoner()

    # with open(f"{task_dir}/result.pkl", "wb") as f:
    #     pickle.dump(plan_result, f)

    env.close()

    end = time.time()

    with open(f"{task_dir}/time.txt", "a+") as f:
        f.write(f"total time taken: {end - start}\n")

    return False


tasks_0 = [
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
]

tasks_1 = [
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
]

tasks_2 = [
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

tasks_3 = [
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
]

tasks_4 = [
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
]

tasks_5 = [
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


if portion == 0:
    run_exp(name, tasks_0)
elif portion == 1:
    run_exp(name, tasks_1)
elif portion == 2:
    run_exp(name, tasks_2)
elif portion == 3:
    run_exp(name, tasks_3)
elif portion == 4:
    run_exp(name, tasks_4)
elif portion == 5:
    run_exp(name, tasks_5)


"""
<TimeLimit<OrderEnforcing<BrowserEnv<browsergym/webarena.30>>>>
0
False False
1
False False
2
False False
3
False False
4
False False
5
False False
6
False False
7
False False
8
False False
9
False False
10
False False
11
False False
12
False False
13
False False
14
False False
15
False False
16
False False
17
False False
18
False False
19
False False
20
False False
21
False False
22
False False
23
False False
24
False False
25
False False
26
False False
27
False False
28
False False
29
False False
30
False False
31
False False
32
False False
33
False False
34
False False
35
False False
36
False False
37
False False
38
False False
39
False False
40
False False
41
False False
42
False False
43
False False
44
False False
45
False False
46
False False
47
False False
48
False False
49
False False
50
False False
51
False False
52
False False
53
False False
54
False False
55
False False
56
False False
57
False False
58
False False
59
False False
60
False False
61
False False
62
False False
63
False False
64
False False
65
False False
66
False False
67
False False
68
False False
69
False False
70
False False
71
False False
72
False False
73
False False
74
False False
75
False False
76
False False
77
False False
78
False False
79
False False
80
False False
81
False False
82
False False
83
False False
84
False False
85
False False
86
False False
87
False False
88
False False
89
False False
90
False False
91
False False
92
False False
93
False False
94
False False
95
False False
96
False False
97
False False
98
False False
99
False True
"""

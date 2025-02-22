import math
import sys
import argparse
import os
# import numpy as np

wa_reddit_tasks = [
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


def get_tag_prev_exps_count(results_dir: str, tag: str) -> int:
  items = os.listdir(results_dir)
  exp_names = list(sorted(filter(lambda name: tag in name, items)))
  return len(exp_names)

def filter_out_completed_tasks(results_dir: str, tag: str, tasks: list[str] = wa_reddit_tasks):
  items = os.listdir(results_dir)
  exp_names = list(sorted(filter(lambda name: tag in name, items)))

  combined_status = ""
  for exp_name in exp_names:
    exp_path = os.path.join(results_dir, exp_name)
    status = open(os.path.join(exp_path, "status.txt")).read()
    combined_status += status + "\n"

  filtered_tasks = []
  for task in tasks:
    if f"{task} True" not in combined_status and f"{task} False" not in combined_status:
      filtered_tasks.append(task)
  return filtered_tasks

def get_tasks_subset_for_portion(total_portions: int, portion: int, tasks: list[str] = wa_reddit_tasks):
  portion_size = math.ceil(len(tasks) / total_portions)
  start_idx = (portion-1) * portion_size
  end_idx = start_idx + portion_size
  return tasks[start_idx:end_idx]

# parser = argparse.ArgumentParser()
# parser.add_argument("--total-portions")
# parser.add_argument("--portion-idx")
# parser.add_argument("--mcts-iterations")
# parser.add_argument("--mcts-depth")
# args = parser.parse_args()
# print(args)
# total_portions = int(sys.argv[1])
# portion = int(sys.argv[2])

# print(get_tasks_for_portion(total_portions, portion))

# result = []
# for idx in range(10):
#   result += get_tasks_for_portion(10, idx)

# print(np.all(np.array(result) == np.array(tasks)))
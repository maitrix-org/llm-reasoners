import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import pickle
from itertools import chain

def get_stats(tag: str):
  success_stats = get_success_stats(tag)
  token_stats = get_token_stats(tag)
  time_stats = get_time_stats(tag)
  iteration_stats = get_mcts_stats(tag)
  stats = list(chain(success_stats, token_stats, time_stats, iteration_stats))
  return stats


def get_success_stats(tag: str):
  successes = 0
  failures = 0
  errors = 0

  results_dir = "./results"
  exp_names = os.listdir(results_dir)
  for exp_name in exp_names:
    if tag in exp_name:

      # aggregate task success stats from status.txt
      with open(os.path.join(results_dir, exp_name, "status.txt")) as f:
        for line in f.readlines():
          if line.startswith("webarena"):
            status = line.strip().split(" ")[1]
            # print(status)
            if status == "True":
              successes += 1
            elif status == "False":
              failures += 1
            else:
              errors += 1
          else: # otherwise it's error output or something. can ignore. 
            pass
  return [int(tag[1:]), successes, failures, errors]


def get_token_stats(tag: str):
  task_token_stats_rows = []
  results_dir = "./results"
  exp_names = os.listdir(results_dir)
  for exp_name in exp_names:
    if tag in exp_name:

      exp_path = os.path.join(results_dir, exp_name)
      task_names = os.listdir(exp_path)

      for task_name in task_names:
        if "webarena" in task_name:

          calls_made = 0
          prompt_tokens = 0
          cached_prompt_tokens = 0
          completion_tokens = 0

          task_path = os.path.join(results_dir, exp_name, task_name)
          items = os.listdir(task_path)
          pickles = [x for x in items if x.endswith(".pkl")]
          for file in pickles:
            if file != "result.pkl":
              pickle_path = os.path.join(task_path, file)
              response = pickle.load(open(pickle_path, "rb"))
              calls_made += 1
              prompt_tokens += response.usage.prompt_tokens
              cached_prompt_tokens += response.usage.prompt_tokens_details.cached_tokens
              completion_tokens += response.usage.completion_tokens

          task_token_stats_rows.append(
            [task_name, calls_made, prompt_tokens, cached_prompt_tokens, completion_tokens]
          )
  
  task_token_df = pd.DataFrame(task_token_stats_rows,
                             columns=["task_name", "llm_calls_made", "prompt_tokens", "prompt_cached_tokens", "completion_tokens"])
  task_token_df["total_tokens"] = task_token_df["prompt_tokens"] + task_token_df["completion_tokens"]
  task_token_df["prompt_non_cached_tokens"] = task_token_df["prompt_tokens"] - task_token_df["prompt_cached_tokens"]

  gpt_4o_mini_input_ppmt = 0.150
  gpt_4o_mini_cachedinput_ppmt = 0.075
  gpt_4o_mini_output_ppmt = 0.600

  task_token_df["usd_cost"] = (task_token_df["prompt_non_cached_tokens"] * gpt_4o_mini_input_ppmt / 1e6 + 
                              task_token_df["prompt_cached_tokens"] * gpt_4o_mini_cachedinput_ppmt / 1e6 + 
                              task_token_df["completion_tokens"] * gpt_4o_mini_output_ppmt / 1e6)
  task_token_df = task_token_df[["task_name", "usd_cost", "prompt_tokens", "prompt_cached_tokens", "prompt_non_cached_tokens", "completion_tokens", "total_tokens"]]
  task_token_df.set_index("task_name")

  total_usd = task_token_df["usd_cost"].sum()
  avg_usd = task_token_df["usd_cost"].mean()

  task_token_summary_stats = task_token_df.describe()
  totals = task_token_df.sum(axis=0)
  totals.name = "sum"
  task_token_summary_stats = pd.concat([task_token_summary_stats, totals.to_frame().T])
  task_token_summary_stats

  return total_usd, avg_usd, task_token_df


def get_time_stats(tag: str):
  task_time_stats_rows = []
  results_dir = "./results"
  exp_names = os.listdir(results_dir)
  for exp_name in exp_names:
    if tag in exp_name:

      exp_path = os.path.join(results_dir, exp_name)
      task_names = os.listdir(exp_path)

      for task_name in task_names:
        if "webarena" in task_name:
          
          total_proposal_time = 0
          total_evaluation_time = 0
          total_envstep_time = 0
          total_time_taken = 0

          task_path = os.path.join(results_dir, exp_name, task_name)
          with open(os.path.join(task_path, "time.txt")) as f:
            for line in f.readlines():

              if line.startswith("action proposal time"):
                total_proposal_time += float(line.strip().split(": ")[1])
              elif line.startswith("total action evaluation time"):
                total_evaluation_time += float(line.strip().split(": ")[1])
              elif line.startswith("env step time"):
                total_envstep_time += float(line.strip().split(": ")[1])
              elif line.startswith("total time taken"):
                total_time_taken = float(line.strip().split(": ")[1])
          
          task_time_stats_rows.append([task_name, total_time_taken, total_proposal_time, total_evaluation_time, total_envstep_time])
  task_time_df = pd.DataFrame(task_time_stats_rows,
                             columns=["task_name", "total_time_taken", "total_proposal_time", "total_evaluation_time", "total_envstep_time"])
  
  total_time = task_time_df["total_time_taken"].sum()
  avg_time = task_time_df["total_time_taken"].mean()

  return total_time, avg_time, task_time_df


def get_mcts_stats(tag: str):
  task_mcts_stats_rows = []
  results_dir = "./results"
  exp_names = os.listdir(results_dir)
  for exp_name in exp_names:
    if tag in exp_name:

      exp_path = os.path.join(results_dir, exp_name)
      task_names = os.listdir(exp_path)

      for task_name in task_names:
        if "webarena" in task_name:
          task_path = os.path.join(results_dir, exp_name, task_name)
          if os.path.exists(os.path.join(task_path, "result.pkl")):
            mcts_result = pickle.load(open(os.path.join(task_path, "result.pkl"), "rb"))
            completion_iteration = np.nan
            completion_depth = np.nan
            if mcts_result.cum_reward >= 100: # task successfully completed
              completion_iteration = find_completion_iteration(mcts_result)
              completion_depth = find_completion_depth(mcts_result)
            max_depth = find_max_depth(mcts_result)
            env_steps_taken = get_env_steps_taken(mcts_result)
            task_mcts_stats_rows.append([task_name, completion_iteration, completion_depth, max_depth, env_steps_taken])
            # else:
            #   task_mcts_stats_rows.append([task_name, np.nan, np.nan])
          else:
            task_mcts_stats_rows.append([task_name, np.nan, np.nan])
  
  task_mcts_df = pd.DataFrame(task_mcts_stats_rows,
                             columns=["task_name", "completion_iteration", "completion_depth", "max_depth", "env_steps_taken"])
  task_mcts_df.set_index("task_name", inplace=True)
  avg_completion_iteration = task_mcts_df["completion_iteration"].mean()
  avg_completion_depth = task_mcts_df["completion_depth"].mean()

  return avg_completion_iteration, avg_completion_depth, task_mcts_df


# assumes mcts_result passed in is for a task that has been completed successfully
def find_completion_iteration(mcts_result):
  last_trace = mcts_result.trace_in_each_iter[-1]
  same_info = []
  for idx, trace in enumerate(mcts_result.trace_in_each_iter):
    same_trace = True
    for n1, n2 in zip(trace, last_trace):
      if n1.id != n2.id:
        same_trace = False
        break
    same_info.append(same_trace)
    # print(f"trace {idx} - {same_trace}")
      
  for idx, si in enumerate(same_info):
    if si:
      # print(f"task completed at iteration {idx}")
      return idx
    
def find_completion_depth(mcts_result):
  return len(mcts_result.trace_in_each_iter[-1])

def find_max_depth(mcts_result):
  return len(max(mcts_result.trace_in_each_iter, key=lambda x: len(x)))

def get_env_steps_taken(mcts_result):
  completion_iteration = find_completion_iteration(mcts_result)
  steps_taken = 0
  for trace in mcts_result.trace_in_each_iter[:completion_iteration+1]:
    steps_taken += len(trace)
  return steps_taken
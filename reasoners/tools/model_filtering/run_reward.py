#!/usr/bin/env python3
"""
Reward-only pass for batched inference with preserved response ordering
"""

# Standard library imports
import argparse
import glob
import json
import os
import signal
import time
from datetime import datetime, timedelta
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import Optional, Dict, List, Tuple, Any, Union

# Third party imports
from tqdm import tqdm

# Local imports
from verl.utils.reward_score import _default_compute_score
from model_filtering.utils import console, json_default

# --------------------------------------------------------------------------- #
# Globals set once per worker via Pool.initializer                            #
# --------------------------------------------------------------------------- #
_TASK_TIMEOUT = None  # seconds


def _init_pool_processes(task_timeout: int):
    """Initializer so every worker knows the hard timeout value."""
    global _TASK_TIMEOUT
    _TASK_TIMEOUT = task_timeout


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _timeout_handler(signum, frame):  # noqa: D401, N802
    """SIGALRM handler that simply raises a TimeoutError."""
    raise TimeoutError("Hard task timeout hit")


def compute_single_reward(arg_tuple):
    """
    Compute reward for one (response, ground-truth) pair.

    * arg_tuple = (gid, response, data_source, ground_truth, extra_info, resp_idx)
    * Returns (gid, detailed_dict, resp_idx)
    """
    gid, response, data_source, ground_truth, extra_info, resp_idx = arg_tuple

    if _TASK_TIMEOUT is not None:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_TASK_TIMEOUT)

    try:
        result = _default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if isinstance(result, dict):
            detailed = result
            score = detailed.get("score", 0.0)
        else:  # float
            detailed = {"score": float(result)}
            score = float(result)
        detailed["score"] = score
        return gid, detailed, resp_idx

    except TimeoutError:
        return gid, {"score": 0.0, "error": f"task_timeout>{_TASK_TIMEOUT}s"}, resp_idx
    except Exception as e:
        return gid, {"score": 0.0, "error": str(e)}, resp_idx
    finally:
        if _TASK_TIMEOUT is not None:
            signal.alarm(0)


# --------------------------------------------------------------------------- #
# Per-rank scoring                                                            #
# --------------------------------------------------------------------------- #
def score_rank_dir(rank_dir: str, args, reward_pool: Optional[Pool]):
    batch_files = (
        [os.path.join(rank_dir, "batch_00000.json")]
        if args.debug
        else sorted(glob.glob(os.path.join(rank_dir, "batch_*.json")))
    )
    if not batch_files:
        console.print(f"⚠️  [warning]No batch files found in {rank_dir} — skipping[/warning]")
        return 0.0

    rank_start = time.time()
    rank_results = {}
    total_responses = 0

    for batch_file in tqdm(batch_files, desc=f"💯 Scoring {os.path.basename(rank_dir)}", position=0):
        with open(batch_file, "r") as f:
            batch = json.load(f)

        tasks, lookup, gid = [], {}, 0
        batch_name = os.path.basename(batch_file)

        # Track the number of responses per sample to pre-allocate results arrays
        response_counts = {}

        for s_idx, sample in batch.items():
            if not args.recalculate_rewards and sample.get("scores"):
                rank_results[f"{batch_name}_{s_idx}"] = sample
                continue

            # Store the count of responses for this sample
            response_counts[s_idx] = len(sample["responses"])

            for resp_idx, raw_resp in enumerate(sample["responses"]):
                stripped = raw_resp.split("</think>", 1)[1] if "</think>" in raw_resp else raw_resp
                # Include the resp_idx in the task tuple
                tasks.append((gid, stripped, sample["source"], sample["ground_truth"], sample["extra_info"], resp_idx))
                lookup[gid] = s_idx  # Maps back to sample index
                gid += 1

        total_responses += len(tasks)

        if tasks:
            results_iter = (
                reward_pool.imap_unordered(compute_single_reward, tasks, chunksize=1)
                if reward_pool
                else map(compute_single_reward, tasks)
            )

            inner_pbar = tqdm(
                total=len(tasks),
                desc=f"🧮 Responses in {batch_name}",
                position=1,
                leave=False,
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
            )

            # Initialize results containers with the right size for each sample
            detailed_by_sample: Dict[str, List[Optional[Dict]]] = {
                s_idx: [None] * count for s_idx, count in response_counts.items()
            }
            
            # Process results as they come in, maintaining order
            for gidx, detailed, resp_idx in results_iter:
                s_idx = lookup[gidx]
                # Place the result at the correct position based on resp_idx
                detailed_by_sample[s_idx][resp_idx] = detailed
                inner_pbar.update(1)
            
            inner_pbar.close()

            for s_idx, sample in batch.items():
                if s_idx not in detailed_by_sample:
                    continue
                
                detailed_list = detailed_by_sample[s_idx]
                
                # Verify all positions were filled
                if None in detailed_list:
                    missing_indices = [i for i, d in enumerate(detailed_list) if d is None]
                    console.print(f"⚠️ Missing results for sample {s_idx} at positions {missing_indices}")
                    # Fill any missing results with error placeholders
                    for idx in missing_indices:
                        detailed_list[idx] = {"score": 0.0, "error": "result_missing"}
                
                scores = [d["score"] for d in detailed_list]
                pass_cnt = sum(s >= args.correct_reward_threshold for s in scores)
                sample.update(
                    {
                        "detailed_scores": detailed_list,
                        "scores": scores,
                        "pass_rate": pass_cnt / len(scores) if scores else 0.0,
                    }
                )
                rank_results[f"{batch_name}_{s_idx}"] = sample
            
            # Calculate and print overall pass rate for this batch
            batch_pass_rates = [sample["pass_rate"] for sample in batch.values() if "pass_rate" in sample]
            if batch_pass_rates:
                avg_batch_pass_rate = sum(batch_pass_rates) / len(batch_pass_rates)
                console.print(f"📊 {batch_name} pass rate: {avg_batch_pass_rate:.2%}")
        else:
            console.print(f"ℹ️  All samples in {batch_name} already scored; skipping computation")

        with open(batch_file, "w") as f:
            json.dump(batch, f, indent=2, default=json_default)

    elapsed = time.time() - rank_start
    final_path = os.path.join(rank_dir, "final_results.json")
    with open(final_path, "w") as f:
        json.dump(
            {
                "results": rank_results,
                "errors": {},
                "metrics": {
                    "total_time": elapsed,
                    "num_responses": total_responses,
                    "avg_reward_time": elapsed / max(1, total_responses),
                },
            },
            f,
            indent=2,
            default=json_default,
        )
    console.print(f"✅ Saved summary to [highlight]{final_path}[/highlight]")
    return elapsed


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Reward-only pass (efficient, balanced) with preserved ordering")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_parquet_path", type=str, required=True,
                        help="Only used to locate the dataset-named output directory")
    parser.add_argument("--output_dir", type=str, default="./diff_filter_output")

    parser.add_argument("--reward_workers", type=int, default=16,
                        help="Upper bound on CPU processes; auto-downscales.")
    parser.add_argument("--correct_reward_threshold", type=float, default=1.0)
    parser.add_argument("--recalculate_rewards", action="store_true",
                        help="Recompute even if sample already has scores")
    parser.add_argument("--debug", action="store_true",
                        help="Process only batch_00000.json per rank")

    parser.add_argument("--task_timeout", type=int, default=35, # a little longer than 30s timeout of code_exec_local
                        help="Hard wall-clock timeout per response (s)")
    parser.add_argument("--maxtasks_per_child", type=int, default=100,
                        help="Recycle worker after N tasks")
    parser.add_argument("--join_timeout", type=int, default=30,
                        help="Max seconds to wait for pool shutdown before force-kill")

    args = parser.parse_args()

    avail = cpu_count() or 1
    reserved = max(1, avail // 8)
    workers = (
        1 if args.reward_workers <= 1 else min(args.reward_workers, max(1, avail - reserved))
    )

    console.rule("[bold]Difficulty Filter — Ordered Reward pass", style="cyan")
    console.print(f"⏰  Start : {datetime.now():%Y-%m-%d %H:%M:%S}")
    console.print(f"🖥️  CPUs  : available={avail}, using={workers}")
    console.print(f"⏱️  Hard per-response timeout : {args.task_timeout}s")
    console.rule(style="cyan")

    dataset_name = os.path.basename(args.dataset_parquet_path).rsplit(".parquet", 1)[0]
    model_name = args.model_path.split("/")[-1]
    root_dir = os.path.join(args.output_dir, dataset_name, model_name)

    rank_dirs = sorted(glob.glob(os.path.join(root_dir, "dp*")))
    if not rank_dirs:
        console.print(f"❌ [error]No dp* directories under {root_dir}")
        return

    reward_pool = (
        Pool(
            processes=workers,
            initializer=_init_pool_processes,
            initargs=(args.task_timeout,),
            maxtasksperchild=args.maxtasks_per_child,
        )
        if workers > 1
        else None
    )

    total_elapsed = 0.0
    try:
        for rd in rank_dirs:
            total_elapsed += score_rank_dir(rd, args, reward_pool)
    finally:
        if reward_pool:
            reward_pool.close()

            deadline = time.time() + args.join_timeout
            while any(p.is_alive() for p in reward_pool._pool) and time.time() < deadline:
                time.sleep(0.25)

            if any(p.is_alive() for p in reward_pool._pool):
                console.print("[yellow]Pool shutdown took too long — forcing terminate()[/yellow]")
                reward_pool.terminate()

            reward_pool.join()

    console.rule(style="cyan")
    console.print(f"🏁 Finished scoring in {format_time(total_elapsed)}")


if __name__ == "__main__":
    main()
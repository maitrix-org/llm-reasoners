#!/usr/bin/env python3
import argparse
import os
import sys
import signal
import multiprocessing
from multiprocessing import Process
import torch
import vllm
from vllm.utils import get_open_port
from datetime import datetime
from rich.panel import Panel

from model_filtering.utils import console
from model_filtering.pipeline import DifficultyFilterPipeline
from datasets import Dataset

# --------------------------------------------------------------------------- #
# Data-parallel worker                                                        #
# --------------------------------------------------------------------------- #
def run_dp_worker(args, dp_rank, dp_size, dataset):

    os.environ.update(
        {
            "VLLM_DP_RANK": str(dp_rank),
            "VLLM_DP_RANK_LOCAL": str(dp_rank),
            "VLLM_DP_SIZE": str(dp_size),
            "VLLM_DP_MASTER_IP": args.master_addr,
            "VLLM_DP_MASTER_PORT": str(args.master_port),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_TORCH_DIST": "1",
        }
    )
    console.print(f"[DP rank {dp_rank}] sees {torch.cuda.device_count()} visible GPU(s).")


    args.dp_rank = dp_rank
    args.dp_size = dp_size

    console.rule(f"[bold]Worker Configuration DP {dp_rank}/{dp_size-1}", style="cyan")
    console.print(
        Panel(
            f"[bold]Model:[/bold] {args.model_path}\n"
            f"[bold]Dataset:[/bold] {args.dataset_parquet_path}\n"
            f"[bold]Batch size:[/bold] {args.batch_size}\n"
            f"[bold]Generations:[/bold] {args.n}\n"
            f"[bold]Max tokens:[/bold] {args.max_new_tokens}\n"
            f"[bold]Tensor parallel:[/bold] {args.tp_size}\n"
            f"[bold]GPU Device:[/bold] {torch.cuda.current_device()}",
            title="📋 Configuration",
            border_style="cyan",
        )
    )

    # ---------- Inference only (no reward) --------------------------------- #
    pipeline = DifficultyFilterPipeline(args, dataset)
    pipeline.run_inference()

# --------------------------------------------------------------------------- #
# CLI / launcher                                                              #
# --------------------------------------------------------------------------- #
def main():
    def handle_signal(signum, _):
        console.print(f"\n⚠️ [warning]Received signal {signum}, shutting down…[/warning]")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(description="Inference-only pipeline (rewards run later)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_parquet_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./diff_filter_output")

    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument(
        "--truncation", type=str, default="error", choices=["left", "right", "error"]
    )
    parser.add_argument("--default_data_source", type=str, default="None")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--enable_expert_parallel", action="store_true",
                        help="Enable expert parallel mode for model initialization (default: True)")

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--force_regenerate", action="store_true",
                        help="Force regeneration of outputs for all data, ignoring previously saved results")

    parser.add_argument("--skip_percentage", type=float, default=0.0,
                        help="Skip to this percentage (0.0-1.0) of the dataset (e.g., 0.6 for 60%)")

    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--node_size", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=0)

    args = parser.parse_args()

    console.rule("[bold]Difficulty Filter — Inference-only", style="cyan")
    console.print(f"⏰ Start time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    console.rule(style="cyan")
    

    os.makedirs(args.output_dir, exist_ok=True)
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if args.node_size == 1:
        args.master_port = get_open_port()
    else:
        assert args.master_addr != "127.0.0.1"
        assert args.master_port > 0

    assert args.dp_size % args.node_size == 0
    dp_per_node = args.dp_size // args.node_size

    #---Load dataset at main process 
    if "livecodebench" in args.dataset_parquet_path:
        import polars as pl
        console.print(f"🐞 [DEBUG] Loading livecodebench dataset using polars")
        pd_data = pl.read_parquet(args.dataset_parquet_path).to_pandas()
        dataset = Dataset.from_pandas(pd_data)
    else:
        dataset = Dataset.from_parquet(args.dataset_parquet_path)

    #---slice the data and dispatch to each dp worker
    if args.dp_size == 1:
        run_dp_worker(args, dp_rank=0, dp_size=1, dataset=dataset)
    else:
        procs = []
        console.print(f"🔄 Starting {dp_per_node} worker(s) on node {args.node_rank}/{args.node_size-1}")
        for local_rank in range(dp_per_node):
            global_rank = args.node_rank * dp_per_node + local_rank

            # ── DP split
            if args.dp_size > 1:
                total = len(dataset)
                per_rank = total // args.dp_size
                start = global_rank * per_rank
                end = start + per_rank if global_rank != args.dp_size - 1 else total
                console.print("start:", start)
                console.print("end:",end)
                dp_dataset = dataset.select(range(start, end))
                console.print(
                    f"🔢 DP rank [highlight]{global_rank}[/highlight] "
                    f"processing [highlight]{len(dp_dataset)}[/highlight] / {total}"
                )
            else:
                console.print(f"📊 Dataset loaded with [highlight]{len(dataset)}[/highlight] samples")

            p = Process(target=run_dp_worker, args=(args, global_rank, args.dp_size, dp_dataset))
            p.start()
            procs.append(p)

        exit_code = 0
        for proc in procs:
            proc.join()
            if proc.exitcode is None:
                print(f"Killing process {proc.pid}")
                proc.kill()
                exit_code = 1
            elif proc.exitcode:
                exit_code = proc.exitcode
        
        exit(exit_code)


if __name__ == "__main__":
    main()
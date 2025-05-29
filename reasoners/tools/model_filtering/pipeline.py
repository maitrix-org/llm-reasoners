#!/usr/bin/env python3

# Standard library imports
import datetime
import json
import os
import time
import traceback
from datetime import timedelta

# Third party imports 
import numpy as np
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import traceback
# Local imports
from model_filtering.utils import console, json_default, custom_collate_fn

class DifficultyFilterPipeline:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.tokenizer = None
        self.model = None
        self.sampling_params = None

        self.start_time = time.time()
        self.gen_times = []
        self.args.skip_percentage = getattr(self.args, 'skip_percentage', 0.0)

    @staticmethod
    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    # ------------- component init ------------------------------------------ #
    def initialize_components(self):
        console.print(f"🔄 Loading tokenizer from [highlight]{self.args.model_path}[/highlight]...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        console.print(
            f"🚀 Initializing model from [highlight]{self.args.model_path}[/highlight]"
            f"with TP={self.args.tp_size}..."
        )
        console.print(f"Warning: enable_expert_parallel is set to {self.args.enable_expert_parallel}")
        console.print("If you are using MOE models, set enable_expert_parallel to True")
        
        self.model = LLM(
            self.args.model_path,
            tensor_parallel_size=self.args.tp_size,
            enable_expert_parallel=self.args.enable_expert_parallel,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_new_tokens,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            repetition_penalty=self.args.repetition_penalty,
            skip_special_tokens=True,
            truncate_prompt_tokens=self.args.max_prompt_length
        )
        console.print("✅ Model initialization [success]complete[/success].")

    # ------------- dataset -------------------------------------------------- #
    def prepare_dataset(self):
        console.print(f"📂 Loading dataset from [highlight]{self.args.dataset_parquet_path}[/highlight]...")
        console.print(f"🐞 [DEBUG] Dataset loaded with [highlight]{len(self.dataset)}[/highlight] samples")

        # TODO: replace None values with empty strings in dataset columns and nested dictionaries
        # TODO: Below sometimes causes stucks in the process
        # # Replace None values with empty strings in dataset columns and nested dictionaries
        # def replace_none_with_empty(obj, max_depth=3):
        #     if obj is None:
        #         return ""
        #     elif isinstance(obj, dict) and max_depth > 0:
        #         return {k: replace_none_with_empty(v, max_depth-1) for k, v in obj.items()}
        #     elif isinstance(obj, list) and max_depth > 0:
        #         return [replace_none_with_empty(item, max_depth-1) for item in obj]
        #     return obj

        # def process_example(example):
        #     return {k: replace_none_with_empty(v) for k, v in example.items()}

        # # Use batched processing with multiple workers
        # dataset = dataset.map(
        #     process_example,
        #     num_proc=min(8, os.cpu_count()),
        #     desc="Processing None values"
        # )
        # console.print("🧹 Replaced None values with empty strings using parallel processing")
            
        # ── debug slice
        if self.args.debug:
            self.dataset = self.dataset.select(range(min(48, len(self.dataset))))
            console.print(f"🐞 [DEBUG] Using first [highlight]{len(self.dataset)}[/highlight] samples")

        # ── KEEP-ONLY columns actually referenced downstream ---------------- #
        required_cols = {
            "prompt",
            "reward_model",
            "apply_chat_template",
            "data_source",
            "extra_info",
        }
        cols_to_drop = [c for c in self.dataset.column_names if c not in required_cols]
        if cols_to_drop:
            self.dataset = self.dataset.remove_columns(cols_to_drop)
            console.print(f"🧹 Dropped {len(cols_to_drop)} column(s) for easier processing: {', '.join(cols_to_drop)}")

    # ------------- output directory / checkpoint handling ------------------ #
    def get_rank_output_dir(self):
        """Get the output directory for the current rank"""
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name, f"dp{self.args.dp_rank}")
        os.makedirs(rank_output_dir, exist_ok=True)
        return rank_output_dir

    def batch_exists(self, batch_idx):
        """Check if a specific batch has already been processed"""
        rank_output_dir = self.get_rank_output_dir()
        batch_file = os.path.join(rank_output_dir, f"batch_{batch_idx:05d}.json")
        return os.path.exists(batch_file) and not self.args.force_regenerate

    def get_processed_batches_count(self):
        """Count the number of already processed batches"""
        rank_output_dir = self.get_rank_output_dir()
        batch_files = [f for f in os.listdir(rank_output_dir) 
                      if f.startswith("batch_") and f.endswith(".json")]
        return len(batch_files)

    # ------------- batch serialization  ------------------------------------ #
    def process_batch_outputs(self, outputs, batch_dict):
        batch_results = {}
        console.print(f"DP{self.args.dp_rank} Begin to process {len(outputs)} samples")

        tokenizer = self.model.get_tokenizer()
        MAX_LEN = self.args.max_prompt_length

        for i in range(len(outputs)):
            messages = batch_dict["prompt"][i]
            text = "".join(m["content"] for m in messages)
            token_len = len(tokenizer(text).input_ids)

            # build extra_info, adding flags and lengths
            extra_info = {k: batch_dict["extra_info"][k][i]
                        for k in batch_dict["extra_info"]}
            extra_info["too_long"] = (token_len > MAX_LEN)
            extra_info["token_length"] = token_len
            try:
                # collect model outputs
                full_responses = [r.text for r in outputs[i].outputs]

                # compute mean response length in tokens
                if full_responses:
                    resp_token_counts = [
                        len(tokenizer(resp).input_ids) for resp in full_responses
                    ]
                    mean_resp_len = sum(resp_token_counts) / len(resp_token_counts)
                else:
                    mean_resp_len = 0.0
                extra_info["response_mean_length"] = mean_resp_len

                data_source  = batch_dict["data_source"][i]
                if "zebra" in self.args.dataset_parquet_path:
                    ground_truth = ""
                else:
                    ground_truth = batch_dict["reward_model"]["ground_truth"][i]
                question     = next((m["content"] for m in messages if m["role"] == "user"),
                                    "No question found")

                batch_results[i] = {
                    "messages":     messages,
                    "question":     question,
                    "ground_truth": ground_truth,
                    "source":       data_source,
                    "responses":    full_responses,
                    "extra_info":   extra_info,
                }

                if token_len > MAX_LEN:
                    console.print(f"⚠️ Sample {i} is over-length ({token_len} tokens); marked in extra_info.")

            except Exception as e:
                console.print(f"❌ Error processing sample {i}: {e}")
                console.print(traceback.format_exc())

        console.print(f"✅ Finished processing. Kept {len(batch_results)} / {len(outputs)} samples.")
        return batch_results

    # ------------- progress ------------------------------------------------- #
    def print_progress_stats(self, idx, total_batches):
        elapsed = time.time() - self.start_time
        eta = "calculating..."
        if idx > 0 and self.gen_times:
            remain = total_batches - idx - 1
            eta_batch = np.mean(self.gen_times[-10:])
            eta = self.format_time(remain * eta_batch)

        console.print()
        console.print(
            Panel(
                f"[bold]Progress: [metric]{idx+1}/{total_batches}[/metric] "
                f"({(idx+1)/total_batches*100:.1f}%)[/bold]",
                title="📊 Summary",
                border_style="cyan",
            )
        )
        console.print(f"⏱️  Elapsed: [time]{self.format_time(elapsed)}[/time] | ETA: [time]{eta}[/time]")
        if self.gen_times:
            console.print(f"⚡ Generation avg (last 10): [metric]{np.mean(self.gen_times[-10:]):.2f}s[/metric]")

    def check_all_ranks_completed(self, base_output_dir):
        """
        Check if all DP ranks have completed their processing.
        
        Args:
            base_output_dir: Base directory for all DP ranks output
            
        Returns:
            tuple: (is_complete, completed_ranks, missing_ranks)
        """
        if self.args.dp_size <= 1:
            return True, [0], []
            
        completed_ranks = []
        missing_ranks = []
        
        for rank in range(self.args.dp_size):
            rank_dir = os.path.join(base_output_dir, f"dp{rank}")
            rank_completion_file = os.path.join(rank_dir, "RANK_COMPLETE")
            if os.path.exists(rank_completion_file):
                completed_ranks.append(rank)
            else:
                missing_ranks.append(rank)
        
        return len(completed_ranks) == self.args.dp_size, completed_ranks, missing_ranks

    def busy_wait_for_all_ranks(self, base_output_dir, dataloader):
        """
        Wait for all DP ranks to complete their processing while keeping the process active.
        Instead of sleeping, this will continue running inference without saving results.
        
        Args:
            base_output_dir: Base directory for all DP ranks output
            dataloader: The dataloader to use for busy waiting
        """
        if self.args.dp_size <= 1:
            return
            
        console.print(f"\n⏳ [DP-{self.args.dp_rank}] Waiting for all DP ranks to complete with busy waiting...\n")
        
        check_interval = 60  # Check status every 60 seconds
        last_check_time = time.time()
        
        # Create an iterator over the dataloader that loops continuously
        busy_iter = iter(dataloader)
        
        while True:
            # Check if all ranks have completed (but not too frequently)
            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                all_ready, completed_ranks, missing_ranks = self.check_all_ranks_completed(base_output_dir)
                last_check_time = current_time
                
                if all_ready:
                    # Create a nice table summarizing all ranks
                    table = Table(title=f"[DP-{self.args.dp_rank}] DP Ranks Completion Status")
                    table.add_column("Rank", style="cyan")
                    table.add_column("Status", style="green")
                    
                    for rank in range(self.args.dp_size):
                        table.add_row(f"DP-{rank}", "✅ Complete")
                    
                    console.print(table)
                    console.print(f"\n✅ [DP-{self.args.dp_rank}] All {self.args.dp_size} DP ranks have completed their work")
                    break
                else:
                    # Create a table showing status of all ranks
                    table = Table(title=f"[DP-{self.args.dp_rank}] DP Ranks Completion Status")
                    table.add_column("Rank", style="cyan")
                    table.add_column("Status", style="green")
                    
                    for rank in range(self.args.dp_size):
                        if rank in completed_ranks:
                            table.add_row(f"DP-{rank}", "✅ Complete")
                        else:
                            table.add_row(f"DP-{rank}", "⏳ Waiting")
                    
                    console.print(table)
                    console.print(f"\n⏳ [DP-{self.args.dp_rank}] Waiting for {len(missing_ranks)} rank(s) to complete... ({len(completed_ranks)}/{self.args.dp_size} done)")
                    console.print(f"   [DP-{self.args.dp_rank}] Missing: {', '.join(f'DP-{r}' for r in missing_ranks)}")
                    console.print(f"   [DP-{self.args.dp_rank}] Continuing to run inference without saving to stay active")
            
            # Do some work to keep the process active
            try:
                batch_dict = next(busy_iter)
            except StopIteration:
                # Reset the iterator if we reach the end
                busy_iter = iter(dataloader)
                batch_dict = next(busy_iter)
                
            # Run inference on a batch but don't save the results
            console.print(f"\n🔄 [DP-{self.args.dp_rank}] Running busy-wait inference to stay active (results won't be saved)...")
            outputs = self.model.chat(
                batch_dict["prompt"],
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            console.print(f"✅ [DP-{self.args.dp_rank}] Completed busy-wait inference")

    # ------------- main loop ------------------------------------------------ #
    def run_inference(self):
        self.initialize_components()
        self.prepare_dataset()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() // max(1, self.args.dp_size)),
            prefetch_factor=8,
            collate_fn=custom_collate_fn
        )

        total_batches = len(dataloader)
        
        # Get the rank output directory
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = self.get_rank_output_dir()
        
        # Determine starting batch index
        if self.args.skip_percentage > 0:
            start_batch_idx = int(total_batches * self.args.skip_percentage)
            console.print(f"⏩ [DP-{self.args.dp_rank}] Skipping to {self.args.skip_percentage*100:.0f}% of batches (batch {start_batch_idx}/{total_batches})")
        else:
            start_batch_idx = 0
            
        # Count already processed batches for logging
        processed_count = self.get_processed_batches_count()
        if processed_count > 0 and not self.args.force_regenerate:
            console.print(f"📋 [DP-{self.args.dp_rank}] Found {processed_count} already processed batches")
        elif self.args.force_regenerate:
            console.print(f"🔄 [DP-{self.args.dp_rank}] [warning]Force regeneration requested - ignoring existing batch files[/warning]")

        progress_bar = tqdm(
            total=len(dataloader),
            initial=start_batch_idx,
            desc=f"🔄 DP{self.args.dp_rank} Processing",
            position=0,
        )
        
        # Skip to the starting batch
        batch_iter = iter(dataloader)
        for _ in range(start_batch_idx):
            try:
                next(batch_iter)
            except StopIteration:
                console.print(f"⚠️ [DP-{self.args.dp_rank}] [warning]Skip index {start_batch_idx} exceeds dataset size[/warning]")
                break

        # Process each batch with smart checkpoint checking
        for idx, batch_dict in enumerate(batch_iter, start=start_batch_idx):
            # Check if this specific batch has already been processed
            if self.batch_exists(idx):
                console.print(f"\n⏩ [DP-{self.args.dp_rank}] Batch {idx} already exists, skipping...")
                progress_bar.update(1)
                continue
            
            # Batch doesn't exist, process it
            batch_size = len(batch_dict["reward_model"]["ground_truth"])
            console.print(f"\n🔄 [DP-{self.args.dp_rank}] Generating for batch {idx}/{len(dataloader)-1} ({batch_size} samples)…")            
            
            # enforce apply_chat_template==True
            if not all(batch_dict.get("apply_chat_template", [True] * batch_size)):
                raise RuntimeError(
                    "Encountered apply_chat_template=False but raw_prompt column is removed. "
                    "Please ensure all samples set apply_chat_template=True."
                )

            gen_start = time.time()
            outputs = self.model.chat(
                    batch_dict["prompt"],
                    sampling_params=self.sampling_params
            )
            
            self.gen_times.append(time.time() - gen_start)
            console.print(f"⏱️  [DP-{self.args.dp_rank}] Generation took [time]{self.gen_times[-1]:.2f}s[/time]")

            # ----------- store outputs ------------------------- #
            batch_results = self.process_batch_outputs(outputs, batch_dict)

            batch_out = os.path.join(rank_output_dir, f"batch_{idx:05d}.json")
            with open(batch_out, "w") as f:
                json.dump(batch_results, f, indent=2, default=json_default)
            console.print(f"💾 [DP-{self.args.dp_rank}] Saved batch results to [highlight]{batch_out}[/highlight]")

            self.print_progress_stats(idx, len(dataloader))
            progress_bar.update(1)
            console.rule(style="cyan")

        progress_bar.close()
        elapsed_total = time.time() - self.start_time

        console.print()
        console.print(Panel(f"[bold][DP-{self.args.dp_rank}] Inference completed!", title="🏁 Finished", border_style="green"))
        console.print(f"⏱️  [DP-{self.args.dp_rank}] Total time: [time]{self.format_time(elapsed_total)}[/time]")
        console.print(
            f"ℹ️ [DP-{self.args.dp_rank}] All per-batch JSON files are ready. "
            "Run the separate reward script to score and assemble final results."
        )
        
        # Create a completion marker for this rank
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        base_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name)
        completion_file = os.path.join(rank_output_dir, "RANK_COMPLETE")
        with open(completion_file, 'w') as f:
            f.write(f"Completed at {datetime.datetime.now().isoformat()}")
        
        console.print(f"✅ [DP-{self.args.dp_rank}] Created completion marker for DP rank {self.args.dp_rank}")
        
        # Use busy waiting instead of idle sleep for waiting for all ranks
        if self.args.dp_size > 1:
            self.busy_wait_for_all_ranks(base_output_dir, dataloader)
            
        console.print(f"🏁 [DP-{self.args.dp_rank}] Process complete!")
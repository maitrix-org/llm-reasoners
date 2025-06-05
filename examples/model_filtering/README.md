# **Difficulty Model Filtering Pipeline**

## Table of Contents

1. [Overview](#overview)
2. [Resource Requirements](#resource-requirements)
3. [Installation and Setup](#installation-and-setup)
4. [Pipeline Workflow & Scripts](#pipeline-workflow--scripts)

   1. [`run_inference.py`](#1-run_inferencepy)
   2. [`run_reward.py`](#2-run_rewardpy)
   3. [`run_add_pr.py`](#3-run_add_prpy)
   4. [`run_filter.py`](#4-run_filterpy)
   5. [`run_sample_and_postprocess.py`](#5-run_sample_and_postprocesspy)
5. [Checkpoint & Resumption](#checkpoint--resumption)

   * [Advanced Checkpoint Options](#advanced-checkpoint-options)
6. [Utility Functions](#utility-functions)

   * [Generating Pass Rate Mappings](#generating-pass-rate-mappings)
   * [Analyzing Results](#analyzing-results)
7. [Notes](#notes)

---

## Overview

This pipeline performs difficulty filtering by combining two stages:

1. **Inference (GPU‐intensive):** Run a large language model (e.g., Qwen2.5-7B-Instruct, Qwen3-30B-A3B) on a Parquet dataset in data-parallel and tensor-parallel modes to generate responses.
2. **Reward Calculation (CPU‐intensive):** Evaluate those responses with reward functions (e.g., pass/fail metrics) to compute pass rates and categorize problem difficulty.

---

## Resource Requirements

* **GPUs:** 8× H200 (140 GB each)
* **CPUs:** ≥ 64 cores recommended for reward evaluation
* **RAM & Disk:** Sufficient for loading large Parquet files and storing intermediate JSON/Parquet outputs

---

## Installation and Setup

1. **Create and activate a conda environment** (recommended):

   ```bash
   conda create -n model_filtering python=3.10 -y
   conda activate model_filtering
   ```

2. **Install required dependencies:**

   The following Python packages are required for the pipeline scripts:

   - `torch` (PyTorch, version >= 1.13 recommended)
   - `vllm==0.8.5` (required version)
   - `verl` (internal/private package, see below)
   - `datasets` (HuggingFace Datasets)
   - `transformers` (HuggingFace Transformers)
   - `numpy`
   - `polars`
   - `pandas`
   - `rich`
   - `tqdm`
   - `matplotlib` (only needed for plotting in some scripts, e.g., test_parquet.py)

   **Install public dependencies with pip:**

   ```bash
   pip install torch
   pip install vllm==0.8.5
   pip install datasets transformers numpy polars pandas rich tqdm matplotlib
   ```

   > **Important:** The pipeline specifically requires `vllm` version 0.8.5. If you have another version installed, uninstall it first:
   >
   > ```bash
   > pip uninstall vllm
   > pip install vllm==0.8.5
   > ```

   **Install `verl`:**
   - `verl` is an internal/private package. Please refer to your organization or project maintainer for installation instructions. Typically, this may involve cloning a private repository and running `pip install -e .` in the `verl` directory.

3. **Verify GPU drivers and distributed setup** (e.g., NCCL, CUDA, networking) for multi-GPU or multi-node execution.

---

## Pipeline Workflow & Scripts


### 1. `run_inference.py`

**Purpose:**

* Load a Parquet dataset of prompts and run model inference in distributed (data-parallel + tensor-parallel) fashion.
* Generate *n* completions per prompt and save outputs as batch-indexed JSON files under the specified output directory.

**Key Arguments:**

* `--model_path`: Model identifier or local path (e.g., `"Qwen/Qwen2.5-7B-Instruct"`).

* `--dataset_parquet_path`: Path to input Parquet (e.g., `"data/train/codegen__leetcode2k_2.4k.parquet"`).

* `--output_dir`: Directory to save inference JSON outputs.

* `--max_prompt_length`: Max tokens for each prompt (e.g., 4096).

* `--truncation`: Truncation strategy if prompt exceeds length (`left`, `right`, `error`).

* `--batch_size`: Batch size per GPU (e.g., 128).

* `--n`: Number of generated completions per prompt (e.g., 16).

* `--max_new_tokens`: Maximum tokens to generate per completion (e.g., 4096).

* Parallel/distributed flags:

  * `--dp_size` (data-parallel world size)
  * `--tp_size` (tensor-parallel world size)
  * `--node_size`, `--node_rank`, `--master_addr`, `--master_port`

* `--force_regenerate` (optional): Ignore all existing batch JSONs in `output_dir` and regenerate from scratch.

**Example A: Qwen2.5-7B-Instruct (\~ 0.09 s per data point on leetcode2k)**

```bash
python model_filtering/run_inference.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --max_prompt_length 4096 \
  --truncation "left" \
  --dp_size 8 \
  --tp_size 1 \
  --node_size 1 \
  --node_rank 0 \
  --master_addr "127.0.0.1" \
  --master_port 0 \
  --batch_size 128 \
  --n 16 \
  --max_new_tokens 4096
```

**Example B: Qwen3-30B-A3B (\~ 12 s per data point on leetcode2k)**

```bash
python model_filtering/run_inference.py \
  --model_path "Qwen/Qwen3-30B-A3B" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --max_prompt_length 4096 \
  --truncation "left" \
  --dp_size 4 \
  --tp_size 2 \
  --node_size 1 \
  --node_rank 0 \
  --master_addr "127.0.0.1" \
  --master_port 0 \
  --batch_size 128 \
  --n 16 \
  --max_new_tokens 32768 \
  --enable_expert_parallel
```

---

### 2. `run_reward.py`

**Purpose:**

* Read the JSON outputs from `run_inference.py` (one JSON per batch).
* Evaluate each generated response using reward functions (e.g., pass/fail, custom scoring) in parallel on CPU.
* Save reward results (e.g., pass rates and any detailed metrics) as batch-indexed JSON files in the same output directory.

**Key Arguments:**

* `--model_path`: Same model identifier used during inference (e.g., `"Qwen/Qwen2.5-7B-Instruct"`).
* `--dataset_parquet_path`: Path to the original input Parquet (for consistency and directory structure).
* `--output_dir`: Directory containing inference JSONs.
* `--reward_workers`: Number of CPU processes for reward computation (e.g., 64).
* `--correct_reward_threshold` (optional): A numeric threshold above which a response counts as correct.
* `--recalculate_rewards` (optional): Force recomputation of all rewards, ignoring existing reward JSONs.
* `--task_timeout` (optional): Timeout (in seconds) for each response's reward computation.

**Example: Qwen2.5-7B-Instruct Reward Calculation**

```bash
python model_filtering/run_reward.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --reward_workers 64
```

---

### 3. `run_add_pr.py`

**Purpose:**

* Annotate an existing Parquet dataset by adding two new columns—`low_pass_rate` and `high_pass_rate`—based on provided JSON mappings.
* Output a new Parquet file that merges original data with pass-rate metadata.

**Key Arguments:**

* `--parquet_in`: Input Parquet file to annotate (e.g., `dataset.parquet`).
* `--low_json`: JSON file mapping `idx → pass_rate` for the low threshold (e.g., `low_pass_rates.json`).
* `--high_json`: JSON file mapping `idx → pass_rate` for the high threshold (e.g., `high_pass_rates.json`).
* `--parquet_out_dir`: Directory where the annotated Parquet file will be saved.

**Example:**

```bash
python model_filtering/run_add_pr.py \
  --parquet_in data/train/codegen__leetcode2k_2.4k.parquet \
  --low_json diff_filter_output/Qwen2.5-7B-Instruct/idx_to_passrate_low.json \
  --high_json diff_filter_output/Qwen2.5-7B-Instruct/idx_to_passrate_high.json \
  --parquet_out_dir data/annotated_output
```

---

### 4. `run_filter.py`

**Purpose:**

* Filter a Parquet dataset (e.g., the one produced by `run_add_pr.py`) using two JSON mappings (low-threshold and high-threshold pass rates).
* Drop rows whose pass rates fall outside specified bounds.
* Save the filtered subset(s) as Parquet—and optionally output a summary JSON of filtered indices.

**Key Arguments:**

* `--parquet_in`: Input Parquet file to filter (e.g., `annotated_dataset.parquet`).
* `--parquet_out_dir`: Directory to save filtered Parquet files.
* `--low_json`: JSON mapping index→pass\_rate for low threshold.
* `--high_json`: JSON mapping index→pass\_rate for high threshold.
* `--low_thresh`: Drop rows whose `low_pass_rate` ≤ this value (e.g., 0.1).
* `--high_thresh`: Drop rows whose `high_pass_rate` ≥ this value (e.g., 0.9).
* `--json_out` (optional): Path to save a JSON summary of which indices were filtered.

**Example:**

```bash
python model_filtering/run_filter.py \
  --parquet_in data/annotated_output/codegen__leetcode2k_2.4k.parquet \
  --parquet_out_dir data/filtered_output \
  --low_json diff_filter_output/Qwen2.5-7B-Instruct/idx_to_passrate_low.json \
  --high_json diff_filter_output/Qwen2.5-7B-Instruct/idx_to_passrate_high.json \
  --low_thresh 0.1 \
  --high_thresh 0.9 \
  --json_out data/filtered_output/filtered_summary.json
```

---

### 5. `run_sample_and_postprocess.py`

**Purpose:**

* Take raw Parquet files (possibly across multiple domains) and apply domain-specific filtering, patching, and sampling to reach a target sample size.
* Optionally enforce a maximum prompt length (in tokens) to filter out overly long examples.

**Key Arguments:**

* `--input_data_dir`: Directory containing one or more raw Parquet files (e.g., `data/raw_parquets/`).
* `--input_data_names`: Space-separated list of dataset names (without the `.parquet` extension).
* `--output_data_dir`: Directory where the sampled/filtered Parquet files will be saved.
* `--target_sample_size`: Target number of total examples to select (e.g., 15000).
* `--domain`: Domain type (`math`, `codegen`, `simulation`, `logic`, `table`, or `stem`).
* `--max_prompt_tokens` (optional): Remove any example whose prompt length (token count) exceeds this threshold (e.g., 4096).

**Example:**

```bash
python model_filtering/run_sample_and_postprocess.py \
  --input_data_dir data/raw_parquets \
  --input_data_names dataset1 dataset2 \
  --output_data_dir data/processed_parquets \
  --target_sample_size 15000 \
  --domain codegen \
  --max_prompt_tokens 4096
```

---

## Checkpoint & Resumption

All scripts that process data in batches (namely `run_inference.py` and `run_reward.py`) automatically checkpoint by writing batch-level JSON files to the specified output directory. If a run is interrupted, the script:

1. **Scans** the `output_dir` for existing batch files.
2. **Loads** results from those files.
3. **Continues** from the next unprocessed batch index onward.

This avoids reprocessing already completed work.

### Advanced Checkpoint Options

* **Inference Stage (`run_inference.py`):**

  * `--force_regenerate`: Ignore all existing batch JSON files and start inference from batch 0, overwriting previous outputs. Use this if you change model settings (e.g., temperature) and want fresh generations.

* **Reward Stage (`run_reward.py`):**

  * `--recalculate_rewards`: Force recomputation of all reward scores even if reward JSON files already exist. Use this if you add or modify reward functions and wish to re-evaluate without re-running inference.

---

## Utility Functions

After both stages are complete, use these utilities (in `model_filtering/utils.py`) to generate mappings and analyze difficulty distributions.

---

### Generating Pass Rate Mappings

Creates an `idx_to_passrate.json` for a specific model and dataset—mapping dataset row index → pass rate (0.0–1.0). This is stored under:

```
<output_dir>/<model_name>/idx_to_passrate.json
```

**Usage:**

```bash
python -m model_filtering.utils map \
  --output_dir "./diff_filter_output" \
  --dataset "codegen__leetcode2k_2.4k" \
  --model "Qwen2.5-7B-Instruct"
```

* **Flags:**

  * `--output_dir`: Root output directory containing model subdirectories.
  * `--dataset`: Dataset basename (without `.parquet`).
  * `--model`: Model identifier (subdirectory name).

* **Output:**
  A JSON of the form:

  ```json
  {
    "0": 0.75,
    "1": 0.20,
    // …
  }
  ```

  where each key is the zero-based index of the example in the original Parquet and each value is its pass rate.

---

### Analyzing Results

Categorizes problems into one of seven difficulty buckets based on pass rates. Optionally aggregates across multiple datasets.

**Difficulty Buckets:**

1. **Impossible:** pass rate == 0.0
2. **Very Hard:** (0.0, 0.2)
3. **Hard:** \[0.2, 0.4)
4. **Medium:** \[0.4, 0.6)
5. **Easy:** \[0.6, 0.8)
6. **Very Easy:** (0.8, 1.0)
7. **Perfect:** pass rate == 1.0

**Single‐Dataset Example:**

```bash
python -m model_filtering.utils analyze \
  --output_dir "./diff_filter_output" \
  --datasets "codegen__leetcode2k_2.4k" \
  --model "Qwen2.5-7B-Instruct"
```

**Multi‐Dataset Example (same model):**

```bash
python -m model_filtering.utils analyze \
  --output_dir "./diff_filter_output" \
  --datasets barc_1.1k_chunk_00 barc_1.1k_chunk_01 barc_1.1k_chunk_02 \
  --model "Qwen3-30B-A3B"
```

* **Optional Flags:**

  * `--regenerate`: Force regeneration of `idx_to_passrate.json` mappings before analysis.
  * `--save_combined`: Save a combined mapping and aggregated statistics when analyzing multiple datasets.

* **Outputs:**

  * A summary (to stdout or saved file) listing counts and percentages for each difficulty bucket.
  * If `--save_combined` is used, a combined JSON mapping of all indices across specified datasets.

---

## Notes

* **Python Version:** ≥ 3.8 (recommend ≥ 3.10).
* **Dependencies:** See main `README.md` for full list (e.g., `verl`, `vllm==0.8.5`, `pandas`, `pyarrow`).
* **Distributed Setup:** Ensure NCCL/CUDA networking is correctly configured when using multiple GPUs or nodes.
* **Help Flags:** For detailed options, run:

  ```bash
  python model_filtering/run_inference.py --help
  python model_filtering/run_reward.py --help
  # …etc.
  ```
* **Output Directory Structure Example:**

  ```
  diff_filter_output/
  ├─ Qwen2.5-7B-Instruct/
  │  ├─ inference_batch_000.json
  │  ├─ inference_batch_001.json
  │  ├─ ⋯
  │  ├─ reward_batch_000.json
  │  ├─ reward_batch_001.json
  │  ├─ ⋯
  │  └─ idx_to_passrate.json
  └─ Qwen3-30B-A3B/
     ├─ inference_batch_000.json
     ├─ ⋯
     └─ reward_batch_000.json
  ```
* **Error Recovery:**

  * If a batch JSON is corrupted or missing, manually remove it and rerun the corresponding script with `--force_regenerate` (for inference) or `--recalculate_rewards` (for reward).

---

*End of README*

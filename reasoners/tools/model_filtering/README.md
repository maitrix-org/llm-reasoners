# Difficulty Filter Pipeline

A pipeline for difficulty filtering using reward functions, supporting data-parallel and tensor-parallel processing. The pipeline now operates in two stages:
1. **Inference stage**: Runs the model to generate responses (GPU-intensive)
2. **Reward stage**: Evaluates responses with reward functions (CPU-intensive)

## Resource Requirements

- 8x H200 (140GB) GPUs

## Example Usage

### Stage 1: Inference

> [!IMPORTANT]  
> To run the model, you have to install verl and upgrade vllm to 0.8.5, please first follow the instructions in the main README.md, and then run the following command:

```bash
pip uninstall vllm
pip install vllm==0.8.5
```

#### Qwen2.5-7B-Instruct (~0.09s per data point on leetcode2k)
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

#### Qwen3-30B-A3B (~12s per data point on leetcode2k)

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
### Stage 2: Reward Calculation

After the inference stage is complete, run the reward calculation:

```bash
python model_filtering/run_reward.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --reward_workers 64
```

## Checkpoint & Resumption

The pipeline automatically saves batch results to JSON files in the output directory as it processes data. If a run is interrupted, the pipeline will automatically resume from where it left off by:

1. Scanning the output directory for existing batch files
2. Loading results from these files
3. Continuing from the next unprocessed batch

### Advanced Checkpoint Options

The pipeline provides flags to control checkpoint behavior:

- **Inference stage**:
  - `--force_regenerate`: Ignores all existing checkpoint files and starts processing from the beginning, overwriting previous results

- **Reward stage**:
  - `--recalculate_rewards`: Recalculates all reward scores using the previously generated model responses (useful when implementing new evaluation metrics without re-running the expensive model inference)

## Utility Functions

After running models with data-parallel processing, you can use the utility functions to concatenate and analyze results.

### Generating Pass Rate Mappings

Generate idx to pass_rate mapping for a specific model and dataset:

```bash
python -m model_filtering.utils map \
  --output_dir "./diff_filter_output" \
  --dataset "codegen__leetcode2k_2.4k" \
  --model "Qwen2.5-7B-Instruct"
```

This command will create an `idx_to_passrate.json` file in the model's directory that can be used for analysis.

### Analyzing Results

Analyze difficulty distributions of one or more datasets using a specific model:

```bash
python -m model_filtering.utils analyze \
  --output_dir "./diff_filter_output" \
  --datasets "codegen__leetcode2k_2.4k" \
  --model "Qwen2.5-7B-Instruct"
```

Analyze multiple dataset chunks with the same model:

```bash
python -m model_filtering.utils analyze \
  --output_dir "./diff_filter_output" \
  --datasets barc_1.1k_chunk_00 barc_1.1k_chunk_01 barc_1.1k_chunk_02 \
  --model "Qwen3-30B-A3B"
```

Some advanced options:
- `--regenerate`: Force regeneration of idx_to_passrate mapping
- `--save_combined`: Save the combined mapping when analyzing multiple datasets

The analysis output categorizes problems into seven difficulty levels:
- Impossible (pass rate exactly 0.0)
- Very Hard (pass rate 0.0-0.2, exclusive)
- Hard (pass rate 0.2-0.4)
- Medium (pass rate 0.4-0.6)
- Easy (pass rate 0.6-0.8)
- Very Easy (pass rate 0.8-1.0, exclusive)
- Perfect (pass rate exactly 1.0)


# Annotate Pass Rates Script *run_add_pr.py*

This script adds two new columns to each row of a Parquet dataset, reflecting pass-rate values loaded from two JSON files:

* **`low_pass_rate`**: pass rate from the `--low_json` mapping (default is 0.0 if missing)
* **`high_pass_rate`**: pass rate from the `--high_json` mapping (default is 0.0 if missing)

## Usage

```bash
python run_add_pr.py \
  --parquet_in   <input_dataset>.parquet \
  --low_json     low_pass_rates.json \
  --high_json    high_pass_rates.json \
  --parquet_out_dir  directory for saving processed parquet
```

## Command-Line Arguments

| Argument        | Description                                              | Required |
| --------------- | -------------------------------------------------------- | -------- |
| `--parquet_in`  | Path to the input Parquet file to annotate.              | Yes      |
| `--low_json`    | JSON file mapping `idx → pass_rate` for low thresholds.  | Yes      |
| `--high_json`   | JSON file mapping `idx → pass_rate` for high thresholds. | Yes      |
| `--parquet_out_dir` | Directory where the annotated Parquet file will be saved.     | Yes      |

# Filter by Pass-Rate Thresholds *run_filter.py*

This script filters a Parquet dataset based on pass-rate thresholds defined in two JSON files. It uses **Rich** for a colored, tabular console summary.

## Usage

```bash
python run_filter.py \
  --parquet_in     <input>.parquet  \
  --parquet_out_dir <output_directory>  \
  --low_json       low_pass_rates.json  \
  --high_json      high_pass_rates.json \
  --low_thresh     0.1  \
  --high_thresh    0.9
```

## Arguments

| Flag                | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `--parquet_in`      | Path to the source Parquet file                             |
| `--parquet_out_dir` | Directory where filtered Parquet files will be saved        |
| `--low_json`        | JSON mapping `idx → pass_rate` for low-threshold filtering  |
| `--high_json`       | JSON mapping `idx → pass_rate` for high-threshold filtering |
| `--low_thresh`      | Drop rows with pass rate **≤** this value in low json                  |
| `--high_thresh`     | Drop rows with pass rate **≥** this value in high json                  |

# Math-500 Evaluation

## Overview
This repository contains code for evaluating language models on the [Math-500](https://github.com/hendrycks/math) dataset using SGLang. The evaluation uses two models:
- A policy LLM for generating solutions
- A process reward LLM for evaluating solutions

## Setup

### Prerequisites
1. Install required dependencies:
```bash
pip install latex2sympy2 loguru word2number sglang
```

### Model Setup
1. Start the policy LLM SGLang server (e.g., [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B))
   - Default URL: `http://127.0.0.1:30001`

2. Start the process reward LLM SGLang server (e.g., [`peiyi9979/math-shepherd-mistral-7b-prm`](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm))
   - Default URL: `http://127.0.0.1:30002`

## Usage

Run the evaluation with:
```bash
python run_math500_task.py \
    --reward-sglang-url http://127.0.0.1:30002 \
    --policy-sglang-url http://127.0.0.1:30001 \
    --prompt-path prompts.json \
    --output-path answer.json \
    --beam-size 2 \
    --max-depth 40 \
    --temperature 0.7
```

### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt-path` | Path to prompts JSON file | Required |
| `--output-path` | Path to save results | `answers.json` |
| `--policy-sglang-url` | URL of the policy model | Required |
| `--reward-sglang-url` | URL of the reward model | Required |
| `--beam-size` | Beam size for search | 2 |
| `--max-depth` | Maximum search depth | 40 |
| `--temperature` | Temperature for sampling | 0.7 |
| `--log-file` | Path to log file | `output.log` |

## Results

The evaluation results for the Math-500 dataset are as follows:

### Config Parameters
- Beam Size: 4
- Max Depth: 10
- Number of Actions: 4
- Number of Processes: 128
- Temperature: 0.7

### Model and other details:
- Policy Model: `meta-llama/Llama-3.1-8B-instruct`
- Reward Model: `peiyi9979/math-shepherd-mistral-7b-prm`
- Total Questions Evaluated on: 128

### Metrics
- Total Time: 1042.100 seconds
- Accuracy: 63.4%

## References
- Prompt source: [search-and-learn](https://github.com/huggingface/search-and-learn)
- Evaluation script: [qwen-2.5-math](https://github.com/QwenLM/Qwen2.5-Math)
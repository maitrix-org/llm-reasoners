# Math-500 Evaluation

## Preparation

1. Set up the SGLang server with the primary LLM. The script expects it to run at `http://127.0.0.1:30001/v1` by default.

2. Have the (huggingface) reward model available (path specified via `--reward-model-path`)


## Usage

Run the evaluation script with:

```bash
python run_math500_task.py \
    --reward-model-path /path/to/reward/model \
    --sglang-url http://127.0.0.1:30001/v1 \
    --prompt-path /path/to/prompts.json \
    --output-path answers.json \
    --beam-size 2 \
    --max-depth 40 \
    --temperature 0.7
```

### Command Line Arguments

- `--reward-model-path`: Path to the reward model (required)
- `--prompt-path`: Path to prompts JSON file (required)
- `--output-path`: Path to save results (default: `answers.json`)
- `--sglang-url`: SGLang API URL (default: `http://127.0.0.1:30001/v1`)
- `--reward-model-device`: Device to run reward model on (default: `cuda:0`)
- `--beam-size`: Beam size for search (default: 2)
- `--max-depth`: Maximum search depth (default: 40)
- `--temperature`: Temperature for sampling (default: 0.7)
- `--log-file`: Path to log file (default: `output.log`)


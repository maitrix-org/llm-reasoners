# Math-500 Evaluation

## Preparation

1. Set up the SGLang server with the policy LLM (e.g., [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B)). The script expects it to run at `http://127.0.0.1:30001` by default.
2. Set up the SGLang server with the process reward LLM (e.g., [`peiyi9979/Math-Shepherd`](https://huggingface.co/datasets/peiyi9979/Math-Shepherd)). The script expects it to run at `http://127.0.0.1:30002` by default.


## Usage

Run the evaluation script with:

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

### Command Line Arguments

- `--prompt-path`: Path to prompts JSON file (required)
- `--output-path`: Path to save results (default: `answers.json`)
- `--policy-sglang-url`: Url of the policy model
- `--reward-sglang-url`: Url of the reward model
- `--beam-size`: Beam size for search (default: 2)
- `--max-depth`: Maximum search depth (default: 40)
- `--temperature`: Temperature for sampling (default: 0.7)
- `--log-file`: Path to log file (default: `output.log`)

# ProsQA Evaluation

## Overview
This repository contains code for evaluating reasoning models on the [ProsQA](https://github.com/facebookresearch/coconut/blob/main/data/prosqa_test.json) dataset. 

## Setup

### API Key Setup
- If using the **Deepseek** backend, set the environmental variable:
  ```bash
  export DEEPSEEK_API_KEY=<your_deepseek_api_key>
  ```

- If using the **Openrouter** backend, set the environmental variable:
  ```bash
  export OPENROUTER_API_KEY=<your_openrouter_api_key>
  ```

## Usage

Run the evaluation with:
```bash
python run.py \
    --backend openrouter \
    --model-path deepseek/deepseek-r1 \
    --max_tokens 2048 \
    --temperature 0.0
```

### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--backend` | backend of model, `deepseek` or `openrouter` | `deepseek`|
| `--model-path` | Path to the model | `None` |
| `--max_tokens` | Maximum number of tokens | `None`|
| `--temperature` | Temperature for sampling | `0.0` 

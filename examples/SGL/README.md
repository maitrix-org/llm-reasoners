# Running Examples with SGLang

This directory contains code for running Blocksworld experiments using SGLang. Two reasoning approaches are implemented:

- Chain of Thought (CoT): A basic prompting approach
- Reasoning via Planning (RAP): An advanced planning-based approach from [Hao et al. 2023](https://arxiv.org/abs/2305.14992)

## Setup

1. Make sure you have SGLang server running at `http://127.0.0.1:30001`
2. The code uses Llama-3.1-8B model by default. Make sure you have access to this model.

## Usage

Run the experiments using:

```bash
python blocksworld.py --method <method> [--processes <num_processes>]
```

Arguments:
- `--method`: Required. Choose between:
  - `cot`: Chain of Thought approach
  - `rap`: Reasoning via Planning approach
- `--processes`: Optional. Number of parallel processes to use (default: 100)

## Examples

Run with Chain of Thought:
```bash
python blocksworld.py --method cot
```

Run with Reasoning via Planning:
```bash
python blocksworld.py --method rap --processes 50
```

## Performance Comparison

Average time per case (seconds):

| Framework | CoT | ToT | RAP |
|-----------|-----|-----|-----|
| Previously (HuggingFace) | 2.0 | 125.9 | 129.0 |
| SGLang (Multiprocessing) | 0.066 | - | 1.25 |

Note that the performance of RAP may not be optimal due to hyperparameter settings. To reproduce results in LLM Reasoners [paper](https://arxiv.org/abs/2404.05221), please use the previous [code](examples/RAP/blocksworld/README.md).
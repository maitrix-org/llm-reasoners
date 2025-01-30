# Running Examples with SGLang

This directory contains code for running Blocksworld experiments using SGLang with different reasoning approaches:

- Chain of Thought (CoT)
- Reasoning and Planning (RAP)

## Setup

1. Make sure you have SG-Lang server running at `http://127.0.0.1:30001`

2. The code uses Llama-3.1-8B model by default. Make sure you have access to this model.

## Usage

Run the experiments using:

```bash
python blocksworld_cot.py --method <method> [--processes <num_processes>]
```

Arguments:
- `--method`: Required. Choose between:
  - `cot`: Chain of Thought approach
  - `rap`: Reasoning and Planning approach
- `--processes`: Optional. Number of parallel processes to use (default: 100)

## Examples

Run with Chain of Thought:
```bash
python blocksworld_cot.py --method cot
```

Run with Reasoning via Planning:
```bash
python blocksworld_cot.py --method rap --processes 50
```

## Performance Comparison

Average time per case (seconds):

| Framework | CoT | ToT | RAP |
|-----------|-----|-----|-----|
| Previously (HuggingFace) | 2.0 | 125.9 | 129.0 |
| SGLang (Multiprocessing) | 0.066 | - | 1.25 |
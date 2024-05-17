## Run
An example for exllama (single reasoning chain generation without search)
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/RAP/gsm8k/inference.py --base_lm exllama --exllama_model_dir your/path/to/llama --exllama_lora_dir None --exllama_mem_map '[16,22]' --n_action 1 --n_confidence 1 --n_iters 1 --temperature 0.0
```

An example for llama2 (single reasoning chain generation without search)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/RAP/gsm8k/inference.py --base_lm llama-2 --llama_2_ckpts your/path/to/llama --llama_size "7B"  --n_action 1 --n_confidence 1 --n_iters 1 --temperature 0.0
```

An example running llama3 with RAP (single reasoning chain generation without search)
Please set up `LLAMA3_CKPTS` or change this argument before running.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/RAP/gsm8k/inference.py --base_lm llama-3 --llama_3_ckpts $LLAMA3_CKPTS --llama_size "8B" --n_action 1 --n_confidence 1 --n_iters 1 --temperature 0.0
```





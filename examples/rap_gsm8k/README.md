An example script:

```bash
python examples/rap_gsm8k/inference.py --base_lm exllama --exllama_model_dir $LLAMA2_CKPTS --exllama_lora_dir None --exllama_mem_map '[16,22]' --n_action 1 --n_confidence 1 --n_iters 1 --temperature 0.0
```

An example running llama3 with RAP (single reasoning chain generation without search)

Please set up `LLAMA3_CKPTS` or change this argument before running.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 6666 examples/rap_gsm8k/inference.py --base_lm llama-3 --llama_3_ckpts $LLAMA3_CKPTS --llama_size "8B" --n_action 1 --n_confidence 1 --n_iters 1 --temperature 0.0
```
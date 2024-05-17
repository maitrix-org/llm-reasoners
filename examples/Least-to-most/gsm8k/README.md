## Run

An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/Least-to-most/gsm8k/inference.py \
--base_lm llama-2 \
--llama_2_ckpts "your/path/to/llama" \
--llama_size "7B" \
--temperature 0.0 \
--n_iters 1  
```

An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/Least-to-most/gsm8k/inference.py \
--base_lm llama-3 \
--llama_3_ckpts $LLAMA3_CKPTS \
--llama_size "8B-Instruct" \
--temperature 0.0 \
--n_iters 1  
```


  
## Run
An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/ToT/game24/inference.py --base_lm llama-2 --llama_2_ckpts your/path/to/llama --llama_size "7B"   --batch_size 8 --n_iters 1
```
An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/ToT/game24/inference.py --base_lm llama-3 --llama_3_ckpts $LLAMA3_CKPTS --llama_size "8B-Instruct"    --batch_size 8 --n_iters 1
```

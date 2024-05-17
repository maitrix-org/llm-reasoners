## Run
An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/Least-to-most/strategyQA/inference.py --base_lm llama2  --llama_size "7B"   --temperature 0.0 --llama_ckpts "your/path/to/llama"
```


An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/Least-to-most/strategyQA/inference.py --base_lm llama3  --llama_size "8B-Instruct"    --temperature 0.0 --llama_ckpts $LLAMA3_CKPTS
```

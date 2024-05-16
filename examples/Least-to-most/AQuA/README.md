## Run

An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/Least-to-most/AQuA/inference.py --base_lm llama2 --model_dir your/path/to/llama --llama_size "7B"   --temperature 0.0 
```

An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/Least-to-most/AQuA/inference.py  --base_lm llama3 --model_dir $LLAMA3_CKPTS  --llama_size "8B-Instruct"   --temperature 0.0 
```


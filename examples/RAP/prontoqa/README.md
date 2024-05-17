## Run
An example for exllama
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/RAP/prontoqa/rap_inference.py --base_model exllama --model_dir your/path/to/llama --mem_map "[16, 22]" --depth_limit 6 --n_candidates 1 --temperature 0.0 # | tee debug_rap_chain.log
```

An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234  examples/RAP/prontoqa/rap_inference.py --base_model llama2 --model_dir your/path/to/llama --llama_size "7B"   --temperature 0.0 --n_candidates 1  --depth_limit 6
```

An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234  examples/RAP/prontoqa/rap_inference.py  --base_model llama3 --model_dir $LLAMA3_CKPTS --llama_size "8B-Instruct"  --temperature 0.0 --n_candidates 1  --depth_limit 6
```

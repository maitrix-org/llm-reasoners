## Run

An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 python examples/least_to_most/AQuA_l2m/inference.py --base_lm llama2 --model_dir your/path/to/llama --llama_size "7B"   --temperature 0.0 
```

An example for llama3
```bash
CUDA_VISIBLE_DEVICES=0 python examples/least_to_most/AQuA_l2m/inference.py  --base_lm llama3 --model_dir your/path/to/llama3 --llama_size "8B"   --temperature 0.0 
```

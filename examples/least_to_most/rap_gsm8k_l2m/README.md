## Run
An example for exllama
```bash
CUDA_VISIBLE_DEVICES=0 python examples/least_to_most/rap_gsm8k_l2m/inference.py \
--base_lm exllama \
--exllama_model_dir "your/path/to/llama" \
--exllama_lora_dir None \
--exllama_mem_map None \
--n_iters 1 \
--temperature 0.0 \
--depth_limit 5 \
--n_confidence 1 \
--n_action 1 #| tee least-to-most.log
```

An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 python examples/least_to_most/rap_gsm8k_l2m/inference.py \
--base_lm llama-2 \
--llama_2_ckpts "your/path/to/llama" \
--llama_size "7B" \
--temperature 0.0 \
--n_iters 1  
```
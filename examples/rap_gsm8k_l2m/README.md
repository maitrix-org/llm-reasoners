```bash
CUDA_VISIBLE_DEVICES=4 python examples/rap_gsm8k_l2m/inference.py \
--base_lm exllama \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map None \
--n_iters 1 \
--temperature 0.0 \
--depth_limit 5 \
--n_confidence 1 \
--n_action 1 | tee least-to-most.log
```
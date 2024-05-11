```
CUDA_VISIBLE_DEVICES=0 python examples/reward_LM_gsm8k/inference.py \
    --model_dir /home/rahulc/Desktop/llama/llama-2-7b/quant/ \
    --reward_dir openbmb/Eurus-RM-7b \
    --base_lm exllama \
    --quantized_reward nf4 \
    --reward_lm hf \
    --batch_size 4 \
    --temperature 0.8 \
    --n_sc 4
```

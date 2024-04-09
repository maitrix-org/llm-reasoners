An example script:

```bash
python examples/rap_gsm8k/inference.py --base_lm exllama --exllama_model_dir $LLAMA2_CKPTS --exllama_lora_dir None --exllama_mem_map '[16,22]' --n_action 1 --n_confidence 1 --n_iters 1 --temperature 0.0
```
## Run
For llama2
```bash
CUDA_VISIBLE_DEVICES=0 python examples/CoT/CoT_gsm8k/inference.py --base_lm llama2 --model_dir your/path/to/llama --llama_size "7B"   --batch_size 1  --temperature 0   #| tee CoT_log.log
```
For exllama
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/CoT/CoT_gsm8k/inference.py --base_lm exllama --model_dir your/path/to/llama --lora_dir None --mem_map '[16,22]' --temperature 0  #| tee CoT_log.log
```
## Run
For llama3
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 examples/CoT/gsm8k/inference.py --base_lm llama3 --model_dir $LLAMA3_CKPTS --llama_size "8B"   --batch_size 1  --temperature 0   #| tee CoT_log.log
```
For exllama
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/CoT/gsm8k/inference.py --base_lm exllama --model_dir your/path/to/llama --lora_dir None --mem_map '[16,22]' --temperature 0  #| tee CoT_log.log
```
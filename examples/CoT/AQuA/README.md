## Run
For llama3
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 examples/CoT/AQuA/inference.py --base_lm llama3 --model_dir your/path/to/llama --llama_size "8B"   --batch_size 1  --temperature 0   #| tee cot_log.log
```
For exllama
```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/CoT/AQuA/inference.py --base_lm exllama --model_dir your/path/to/exllama --lora_dir None --mem_map '[16,22]' --temperature 0  #| tee cot_log.log
```

To use other model providers in `reasoners/lm`, simply change the line to load the model in `inference.py`, and change the command lines correspondingly.

 
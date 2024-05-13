## Run
For llama3
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 examples/CoT/strategyQA/inference.py --base_lm llama3 --model_dir $LLAMA3_CKPTS --llama_size "8B"  --batch_size 1  --self_consistency 1 
```



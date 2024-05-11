## Run
For llama2
```bash
CUDA_VISIBLE_DEVICES=0 python examples/cot/cot_strategyQA/inference.py --base_lm llama2 --llama_2_ckpt your/path/to/llama --llama_size "7B"  --batch_size 1  --self_consistency 1 
```



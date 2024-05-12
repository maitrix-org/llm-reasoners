## Run

An example for llama2
First change `fire.Fire(main_exllama)` to `fire.Fire(main)` in `inference.py`
```bash
export LLAMA_2_CKPTS="your/path/to/llama"
CUDA_VISIBLE_DEVICES=0 python examples/least_to_most/least_to_most_strategyQA/inference.py --base_lm llama2  --llama_size "7B"   --temperature 0.0 
```



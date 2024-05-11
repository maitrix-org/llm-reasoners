## Run
For llama2
bfs
```bash
CUDA_VISIBLE_DEVICES=0 python examples/tot/prontoqa/tot_inference.py --base_lm llama2 --model_dir your/path/to/llama --llama_size "7B"   --batch_size 8 --search_algo beam  --log_dir logs/prontoqa_tot_beam --depth_limit 10  --beam_size 10 --temperature 0.8 --reward_aggregator mean 
```
dfs
```bash
CUDA_VISIBLE_DEVICES=0 python examples/tot/prontoqa/tot_inference.py --base_lm llama2 --model_dir your/path/to/llama --llama_size "7B"   --batch_size 8 --search_algo dfs  --log_dir logs/prontoqa_tot_dfs --depth_limit 10 --total_states 10 --temperature 0.8  --max_per_state 3 
```


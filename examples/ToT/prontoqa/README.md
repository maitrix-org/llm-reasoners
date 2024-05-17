## Run
An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
bfs
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/ToT/prontoqa/tot_inference.py --base_lm llama3 --model_dir  $LLAMA3_CKPTS --llama_size "8B-Instruct"   --batch_size 8 --search_algo beam  --log_dir logs/prontoqa_tot_beam --depth_limit 10  --beam_size 10 --temperature 0.8 --reward_aggregator mean 
```
dfs
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 1234 examples/ToT/prontoqa/tot_inference.py --base_lm llama3 --model_dir  $LLAMA3_CKPTS --llama_size "8B-Instruct"   --batch_size 8 --search_algo dfs  --log_dir logs/prontoqa_tot_dfs --depth_limit 10 --total_states 10 --temperature 0.8  --max_per_state 3 
```



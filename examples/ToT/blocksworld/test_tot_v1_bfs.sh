export CUDA_VISIBLE_DEVICES=0
export llama_path="your/path/to/llama3" # or "your/path/to/llama2"
export llama_size="8B" # or "7B" of llama2
export base_lm="llama3" # or "llama2"

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/blocksworld/tot_inference.py --base_lm $base_lm --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_2_data.json'  --depth_limit 2 --model_dir $llama_path --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/bfs_v1_step2_r --beam_size 10 --temperature 0.8 --search_algo beam --reward_aggregator mean --llama_size $llama_size

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/blocksworld/tot_inference.py --base_lm $base_lm --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json'  --depth_limit 4 --model_dir $llama_path --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/bfs_v1_step4_r  --beam_size 10 --temperature 0.8 --search_algo beam --reward_aggregator mean --llama_size $llama_size

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/blocksworld/tot_inference.py --base_lm $base_lm --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_6_data.json'  --depth_limit 6 --model_dir $llama_path --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/bfs_v1_step6_r --beam_size 10 --temperature 0.8 --search_algo beam --reward_aggregator mean --llama_size $llama_size

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/blocksworld/tot_inference.py --base_lm $base_lm --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_8_data.json'  --depth_limit 8 --model_dir $llama_path --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/bfs_v1_step8_r --beam_size 10 --temperature 0.8 --search_algo beam --reward_aggregator mean --llama_size $llama_size

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/blocksworld/tot_inference.py --base_lm $base_lm --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_10_data.json' --depth_limit 10 --model_dir $llama_path --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/bfs_v1_step10_r --beam_size 10 --temperature 0.8 --search_algo beam --reward_aggregator mean --llama_size $llama_size

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/blocksworld/tot_inference.py --base_lm $base_lm --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_12_data.json' --depth_limit 12 --model_dir $llama_path --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/bfs_v1_step12_r --beam_size 10 --temperature 0.8 --search_algo beam --reward_aggregator mean --llama_size $llama_size



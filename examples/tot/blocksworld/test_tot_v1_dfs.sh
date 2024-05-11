
#example for llama2
#export LLAMA2_CKPTS="you/path/to/llama"

CUDA_VISIBLE_DEVICES=0 python examples/tot/blocksworld/tot_inference.py --base_lm llama2 --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_2_data.json'  --depth_limit 2 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step2_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 --llama_size 7B

CUDA_VISIBLE_DEVICES=0 python examples/tot/blocksworld/tot_inference.py --base_lm llama2 --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_4_data.json'  --depth_limit 4 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step4_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 --llama_size 7B

CUDA_VISIBLE_DEVICES=0 python examples/tot/blocksworld/tot_inference.py --base_lm llama2 --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_6_data.json'  --depth_limit 6 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step6_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 --llama_size 7B

CUDA_VISIBLE_DEVICES=0 python examples/tot/blocksworld/tot_inference.py --base_lm llama2 --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_8_data.json'  --depth_limit 8 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step8_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 --llama_size 7B

CUDA_VISIBLE_DEVICES=0 python examples/tot/blocksworld/tot_inference.py --base_lm llama2 --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_10_data.json' --depth_limit 10 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step10_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 --llama_size 7B

CUDA_VISIBLE_DEVICES=0 python examples/tot/blocksworld/tot_inference.py --base_lm llama2 --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_12_data.json' --depth_limit 12 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step12_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 --llama_size 7B



#example for exllama
# CUDA_VISIBLE_DEVICES=0,1 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_2_data.json' --mem_map "[16,22]" --depth_limit 2 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step2_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3

# CUDA_VISIBLE_DEVICES=0,1 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_4_data.json' --mem_map "[16,22]" --depth_limit 4 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step4_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3

# CUDA_VISIBLE_DEVICES=0,1 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_6_data.json' --mem_map "[16,22]" --depth_limit 6 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step6_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3

# CUDA_VISIBLE_DEVICES=0,1 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_8_data.json' --mem_map "[16,22]" --depth_limit 8 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step8_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3

# CUDA_VISIBLE_DEVICES=0,1 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_10_data.json' --mem_map "[16,22]" --depth_limit 10 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step10_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3

# CUDA_VISIBLE_DEVICES=0,1 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_12_data.json' --mem_map "[16,22]" --depth_limit 12 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step12_r --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3
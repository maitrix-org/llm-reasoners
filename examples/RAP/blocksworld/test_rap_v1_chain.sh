# prompt with the same length
# set the depth with the max length

CUDA_VISIBLE_DEVICES=2 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_2_data.json' --depth_limit 2 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step2 --mem_map None --n_iters 1

CUDA_VISIBLE_DEVICES=2 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_4_data.json' --depth_limit 4 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step4 --mem_map None --n_iters 1

CUDA_VISIBLE_DEVICES=2 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_6_data.json' --depth_limit 6 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step6 --mem_map None --n_iters 1

CUDA_VISIBLE_DEVICES=2 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_8_data.json' --depth_limit 8 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step8 --mem_map None --n_iters 1

CUDA_VISIBLE_DEVICES=2 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_10_data.json' --depth_limit 10 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step10 --mem_map None --n_iters 1

CUDA_VISIBLE_DEVICES=2 python examples/blocksworld/rap_inference.py --data_path 'examples/blocksworld/data/split_v1/split_v1_step_12_data.json' --depth_limit 12 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step12 --mem_map None --n_iters 1
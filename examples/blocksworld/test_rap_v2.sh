# prompt with the same length
# set the depth with the max length

CUDA_VISIBLE_DEVICES=2,3 python examples/rap_blocksworld/rap_inference.py --data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_2_data.json' --depth_limit 2 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v2_step_2.json --log_dir logs/v2_step2 --mem_map [16,22]

CUDA_VISIBLE_DEVICES=2,3 python examples/rap_blocksworld/rap_inference.py --data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_4_data.json' --depth_limit 4 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v2_step_4.json --log_dir logs/v2_step4 --mem_map [16,22]

CUDA_VISIBLE_DEVICES=2,3 python examples/rap_blocksworld/rap_inference.py --data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_6_data.json' --depth_limit 6 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v2_step_6.json --log_dir logs/v2_step6 --mem_map [16,22]

CUDA_VISIBLE_DEVICES=2,3 python examples/rap_blocksworld/rap_inference.py --data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_8_data.json' --depth_limit 8 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v2_step_8.json --log_dir logs/v2_step8 --mem_map [16,22]

CUDA_VISIBLE_DEVICES=2,3 python examples/rap_blocksworld/rap_inference.py --data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_10_data.json' --depth_limit 10 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v2_step_10.json --log_dir logs/v2_step10 --mem_map [16,22]

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/rap_inference.py --data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_12_data.json' --depth_limit 12 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v2_step_12.json --log_dir logs/v2_step12 --mem_map [16,22]
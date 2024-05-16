#example for llama2 

export CUDA_VISIBLE_DEVICES=0
export llama_path="your/path/to/llama2"
export llama_size="7B"
python -m torch.distributed.run --nproc_per_node 1 examples/RAP/blocksworld/rap_inference.py --llama_path $llama_path --llama_size $llama_size --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_2_data.json' --depth_limit 2  --batch_size 1 --output_trace_in_each_iter --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step2 
python -m torch.distributed.run --nproc_per_node 1 examples/RAP/blocksworld/rap_inference.py --llama_path $llama_path --llama_size $llama_size --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json' --depth_limit 4  --batch_size 1 --output_trace_in_each_iter --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step4 
python -m torch.distributed.run --nproc_per_node 1 examples/RAP/blocksworld/rap_inference.py --llama_path $llama_path --llama_size $llama_size --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_6_data.json' --depth_limit 6  --batch_size 1 --output_trace_in_each_iter --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step6 
python -m torch.distributed.run --nproc_per_node 1 examples/RAP/blocksworld/rap_inference.py --llama_path $llama_path --llama_size $llama_size --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_8_data.json' --depth_limit 8  --batch_size 1 --output_trace_in_each_iter --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step8 
python -m torch.distributed.run --nproc_per_node 1 examples/RAP/blocksworld/rap_inference.py --llama_path $llama_path --llama_size $llama_size --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_10_data.json' --depth_limit 10  --batch_size 1 --output_trace_in_each_iter --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step10 
python -m torch.distributed.run --nproc_per_node 1 examples/RAP/blocksworld/rap_inference.py --llama_path $llama_path --llama_size $llama_size --data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_12_data.json' --depth_limit 12  --batch_size 1 --output_trace_in_each_iter --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/v1_step12 


            
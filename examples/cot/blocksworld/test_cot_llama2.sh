# export WORLD_SIZE=1
# export MASTER_ADDR=localhost
# export MASTER_PORT=12345
# export RANK=0

export CUDA_VISIBLE_DEVICES=0
export model_dir="llama2"
export log_name="llama2"
export LLAMA_2_CKPTS="your/path/to/llama"
export llama2_size="7B" #7B, 13B, etc.

python examples/cot/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/cot/blocksworld/data/split_v1/split_v1_step_2_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step2/" \
--prompt_path "examples/cot/blocksworld/prompts/pool_prompt_v1.json" \
--llama2_path $LLAMA_2_CKPTS --llama_size $llama2_size \
--temperature 0.0 

python examples/cot/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/cot/blocksworld/data/split_v1/split_v1_step_4_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step4/" \
--prompt_path "examples/cot/blocksworld/prompts/pool_prompt_v1.json" \
--llama2_path $LLAMA_2_CKPTS --llama_size $llama2_size \
--temperature 0.0

python examples/cot/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/cot/blocksworld/data/split_v1/split_v1_step_6_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step6/" \
--prompt_path "examples/cot/blocksworld/prompts/pool_prompt_v1.json" \
--llama2_path $LLAMA_2_CKPTS --llama_size $llama2_size \
--temperature 0.0

python examples/cot/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/cot/blocksworld/data/split_v1/split_v1_step_8_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step8/" \
--prompt_path "examples/cot/blocksworld/prompts/pool_prompt_v1.json" \
--llama2_path $LLAMA_2_CKPTS --llama_size $llama2_size \
--temperature 0.0

python examples/cot/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/cot/blocksworld/data/split_v1/split_v1_step_10_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step10/" \
--prompt_path "examples/cot/blocksworld/prompts/pool_prompt_v1.json" \
--llama2_path $LLAMA_2_CKPTS --llama_size $llama2_size \
--temperature 0.0

python examples/cot/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/cot/blocksworld/data/split_v1/split_v1_step_12_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step12/" \
--prompt_path "examples/cot/blocksworld/prompts/pool_prompt_v1.json" \
--llama2_path $LLAMA_2_CKPTS --llama_size $llama2_size \
--temperature 0.0



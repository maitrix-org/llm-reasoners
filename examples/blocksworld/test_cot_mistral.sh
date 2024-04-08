export CUDA_VISIBLE_DEVICES=1

export model_dir="/data/haotian/RAP_tune/Mistral-7B-v0.1"
export log_name="mistral"
# export model_dir="/data/haotian/RAP_tune/gemma-7b"
# export log_name="gemma"

python examples/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_2_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step2/" \
--prompt_path "examples/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_4_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step4/" \
--prompt_path "examples/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_6_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step6/" \
--prompt_path "examples/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_8_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step8/" \
--prompt_path "examples/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_10_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step10/" \
--prompt_path "examples/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_12_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step12/" \
--prompt_path "examples/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0



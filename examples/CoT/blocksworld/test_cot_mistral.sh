# This script calls huggingface as the backend llm provider.
# You can use other models. Refer to `inference.py` for more details.

export CUDA_VISIBLE_DEVICES=1

export model_dir="/your/path/to/mistral"
export log_name="blocksworld-cot-mistral-log"

# export model_dir="/data/haotian/RAP_tune/gemma-7b"
# export log_name="gemma"

python examples/CoT/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_2_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step2/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/CoT/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step4/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/CoT/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_6_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step6/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/CoT/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_8_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step8/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/CoT/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_10_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step10/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python examples/CoT/blocksworld/cot_inference.py \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_12_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step12/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0
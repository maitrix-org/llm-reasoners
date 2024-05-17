# This script calls huggingface as the backend llm provider.
# You can use other models. Refer to `inference.py` for more details.

export CUDA_VISIBLE_DEVICES=0

export model_dir="llama3"
export log_name="blocksworld-cot-llama3-log"
export llama_path="your/path/to/llama3"
export llama_size="8B-Instruct" # or "8B" etc.


# export model_dir="/data/haotian/RAP_tune/gemma-7b"
# export log_name="gemma"

python -m torch.distributed.run --nproc_per_node 1 examples/CoT/blocksworld/cot_inference.py \
--llama_path  $llama_path \
--llama_size $llama_size \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_2_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step2/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python -m torch.distributed.run --nproc_per_node 1 examples/CoT/blocksworld/cot_inference.py \
--llama_path  $llama_path \
--llama_size $llama_size \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step4/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python -m torch.distributed.run --nproc_per_node 1 examples/CoT/blocksworld/cot_inference.py \
--llama_path  $llama_path \
--llama_size $llama_size \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_6_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step6/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python -m torch.distributed.run --nproc_per_node 1 examples/CoT/blocksworld/cot_inference.py \
--llama_path  $llama_path \
--llama_size $llama_size \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_8_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step8/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python -m torch.distributed.run --nproc_per_node 1 examples/CoT/blocksworld/cot_inference.py \
--llama_path  $llama_path \
--llama_size $llama_size \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_10_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step10/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0

python -m torch.distributed.run --nproc_per_node 1 examples/CoT/blocksworld/cot_inference.py \
--llama_path  $llama_path \
--llama_size $llama_size \
--model_dir $model_dir \
--data_path 'examples/CoT/blocksworld/data/split_v1/split_v1_step_12_data.json' \
--log_dir "logs/${log_name}_blocksworld_cot_v1_step12/" \
--prompt_path "examples/CoT/blocksworld/prompts/pool_prompt_v1.json" \
--temperature 0.0


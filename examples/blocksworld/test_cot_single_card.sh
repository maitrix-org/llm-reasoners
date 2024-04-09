export CUDA_VISIBLE_DEVICES=1

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v2/split_v2_step_2_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v2_step_2.json' \
--log_dir "logs/blocksworld_cot_v2_step2/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v2/split_v2_step_4_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v2_step_4.json' \
--log_dir "logs/blocksworld_cot_v2_step4/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v2/split_v2_step_6_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v2_step_6.json' \
--log_dir "logs/blocksworld_cot_v2_step6/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v2/split_v2_step_8_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v2_step_8.json' \
--log_dir "logs/blocksworld_cot_v2_step8/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v2/split_v2_step_10_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v2_step_10.json' \
--log_dir "logs/blocksworld_cot_v2_step10/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v2/split_v2_step_12_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v2_step_12.json' \
--log_dir "logs/blocksworld_cot_v2_step12/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_2_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/blocksworld_cot_v1_step2/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_4_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/blocksworld_cot_v1_step4/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_6_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/blocksworld_cot_v1_step6/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_8_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/blocksworld_cot_v1_step8/" \
--temperature 0.0

python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_10_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/blocksworld_cot_v1_step10/" \
--temperature 0.0


python examples/blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--data_path 'examples/blocksworld/data/split_v1/split_v1_step_12_data.json' \
--prompt_path 'examples/blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/blocksworld_cot_v1_step12/" \
--temperature 0.0


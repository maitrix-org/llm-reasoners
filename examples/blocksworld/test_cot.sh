CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_2_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v2_step_2.json' \
--log_dir "logs/rap_blocksworld_cot_v2_step2/"

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_4_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v2_step_4.json' \
--log_dir "logs/rap_blocksworld_cot_v2_step4/"

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_6_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v2_step_6.json' \
--log_dir "logs/rap_blocksworld_cot_v2_step6/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_8_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v2_step_8.json' \
--log_dir "logs/rap_blocksworld_cot_v2_step8/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_10_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v2_step_10.json' \
--log_dir "logs/rap_blocksworld_cot_v2_step10/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_12_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v2_step_12.json' \
--log_dir "logs/rap_blocksworld_cot_v2_step12/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v1/split_v1_step_2_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/rap_blocksworld_cot_v1_step2/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v1/split_v1_step_4_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/rap_blocksworld_cot_v1_step4/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v1/split_v1_step_6_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/rap_blocksworld_cot_v1_step6/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v1/split_v1_step_8_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/rap_blocksworld_cot_v1_step8/" \
--temperature 0.0

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v1/split_v1_step_10_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/rap_blocksworld_cot_v1_step10/" \
--temperature 0.0


CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/cot_inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--data_path 'examples/rap_blocksworld/data/split_v1/split_v1_step_12_data.json' \
--prompt_path 'examples/rap_blocksworld/prompts/pool_prompt_v1.json' \
--log_dir "logs/rap_blocksworld_cot_v1_step12/" \
--temperature 0.0


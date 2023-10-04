CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/inference_new.py --data_path 'examples/rap_blocksworld/data/new_data/step_4.json' --depth_limit 4 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v0.json --mem_map [16,22] | tee log_2.log

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 examples/rap_blocksworld/inference_new.py --data_path 'examples/rap_blocksworld/data/new_data/step_4.json' --depth_limit 4 --llama_size 30B --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v0.json | tee log_llama30b.log

    def llama_main(llama_size: str = '13B',
             prompt_path: str = 'examples/rap_blocksworld/prompts/prompt.json',
             data_path: str = 'examples/rap_blocksworld/data/step_4.json',
             disable_log: bool = False,
             config_file: str = "examples/rap_blocksworld/data/bw_config.yaml",
             domain_file: str = "examples/rap_blocksworld/data/generated_domain.pddl",
             lm_plan_file: str = 'lm_plan.tmp',
             depth_limit: int = 6,
             **kwargs):

v2: 2,4,6,8,10,12 (14,16 dicarded because there are only 1 example for each)
num_examples (before extracting prompts)
47
86
155
153
113
46
(-10 as the prompt)

CUDA_VISIBLE_DEVICES=4,5 python examples/rap_blocksworld/inference_new.py --data_path 'examples/rap_blocksworld/data/split_v2/split_v2_step_4_data.json' --depth_limit 4 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter --prompt_path examples/rap_blocksworld/prompts/pool_prompt_v2_step_4.json --log_dir logs/v2_step4 --mem_map [16,22]
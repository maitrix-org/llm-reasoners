CUDA_VISIBLE_DEVICES=0,1 python examples/rap_blocksworld/inference.py --data_path 'full.json' --depth_limit 6 --model_dir $LLAMA2_CKPTS --lora_dir None --batch_size 1 --output_trace_in_each_iter


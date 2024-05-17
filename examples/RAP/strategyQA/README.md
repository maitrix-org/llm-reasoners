## Run
An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0  torchrun --nproc-per-node 1 --master-port 1234  examples/RAP/strategyQA/inference.py --base_lm llama2 --llama_2_ckpt your/path/to/llama --llama_size "7B"   --max_seq_len 4096
```

An example for llama3
Please set up `LLAMA3_CKPTS` or change this argument before running.
```bash
CUDA_VISIBLE_DEVICES=0   torchrun --nproc-per-node 1 --master-port 1234  examples/RAP/strategyQA/inference.py --base_lm llama3 --llama_3_ckpt $LLAMA3_CKPTS --llama_size "8B-Instruct"   --max_seq_len 4096
```




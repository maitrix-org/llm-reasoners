## Run
For llama3
```bash
export PYTHONPATH=your/path/to/llm-reasoners:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 examples/ReAct/hotpotqa/inference.py --base_lm llama3 --model_dir $LLAMA3_CKPTS --llama_size "8B" --batch_size 1 --temperature 0
```
An example running llama3 with ReAct
Please set up `LLAMA3_CKPTS` or change this argument before running.
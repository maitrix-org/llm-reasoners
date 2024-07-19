## Run
For llama3
```bash
export PYTHONPATH=your/path/to/llm-reasoners:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 examples/ReAct/hotpotqa/inference.py --base_lm llama3 --model_dir $LLAMA3_CKPTS --llama_size "8B" --batch_size 1 --temperature 0
```
An example running llama3 with ReAct
Please set up `LLAMA3_CKPTS` or change this argument before running.

I have tested the results of the first 1000 examples from the HotpotQA test set by calculating accuracy using EM and the results are shown in the tables below.
### simplest direct results:
| Model          | Shots   | Accuracy |
|----------------|---------|----------|
| Llama3-8B      | 3-shots |   13.2%  |
| Llama3-8B-Inst | 3-shots |   13.7%  |

### search tool results:
| Model          | Shots   | Accuracy |
|----------------|---------|----------|
| Llama3-8B      | 3-shots |   15.0%  |
| Llama3-8B      | 4-shots |   17.0%  |
| Llama3-8B      | 5-shots |   21.9%  |
| Llama3-8B      | 6-shots |   17.9%  |
| Llama3-8B-Inst | 3-shots |   20.6%  |
| Llama3-8B-Inst | 4-shots |   27.0%  |
| Llama3-8B-Inst | 5-shots |   27.8%  |
| Llama3-8B-Inst | 6-shots |   29.1%  |
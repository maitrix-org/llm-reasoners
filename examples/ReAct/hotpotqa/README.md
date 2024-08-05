# ReAct

This is an example of using Tools to perform Greddy sampling with Llama-3-8B as the base model on the HotpotQA dataset.

## Running the example

Prerequisites:
- Download LLama-3 8B model.
- Have 1 * 24 GB GPU.

Script:
```bash
export PYTHONPATH=your/path/to/llm-reasoners:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node 1 examples/ReAct/hotpotqa/inference.py --base_lm llama3 --model_dir $LLAMA3_CKPTS --llama_size "8B" --batch_size 1 --temperature 0
```

## Results

We tested the results of the first 1000 examples from the HotpotQA test set by calculating accuracy using EM and the results are shown in the tables below.
|Method|Accuracy|
|-|-|
|Without Tools (Llama3-8B)|13.2%|
|Without Tools (Llama3-8B-Inst)|13.7%|
|Without Tools (Llama3.1-8B)|10.5%|
|Without Tools (Llama3.1-8B-Inst)|19.7%|
|With Tools (Llama3-8B)|21.2%|
|With Tools (Llama3-8B-Inst)|29.1%|
|With Tools (Llama3.1-8B)|21.1%|
|With Tools (Llama3.1-8B-Inst)|30.5%|
 

## Reference
```bibtex
@inproceedings{yao2023react,
  title = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  author = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle = {International Conference on Learning Representations (ICLR) },
  year = {2023},
  html = {https://arxiv.org/abs/2210.03629},
}
```
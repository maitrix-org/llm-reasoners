# Experiments
## GSM8K
Running RAP on GSM8K [1]
### RAP
```bash
python -m torch.distributed.run --nproc_per_node 2 examples/rap_gsm8k/inference.py --llama_size "13B"
```
### PAL + Guided Beam Search
```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/guided_gsm8k/inference.py 
```


## Blocksworld
### Preparation
1. Download the validator from [here]()

### RAP
```bash

CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node 4 examples/rap_blocksworld/inference.py --llama_size "30B" --data_path 'examples/blocksworld/data/step_6.json' --depth_limit 6
```


## References
[1] Cobbe, Karl, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert et al. "Training verifiers to solve math word problems." arXiv preprint arXiv:2110.14168 (2021).
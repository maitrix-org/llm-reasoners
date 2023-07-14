# Experiments
## GSM8K
Running RAP on GSM8K [1]
### RAP

### PAL + Guided Beam Search
```bash
```


## Blocksworld
### Preparation
1. Download the validator from [here]()

### RAP
```bash

CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.run --nproc_per_node 4 examples/blocksworld/inference.py --llama_size "30B" --data_path 'examples/blocksworld/data/step_6.json' --depth_limit 6
```


## References
[1] Cobbe, Karl, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert et al. "Training verifiers to solve math word problems." arXiv preprint arXiv:2110.14168 (2021).
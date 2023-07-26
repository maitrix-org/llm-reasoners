# Experiments
## GSM8K
Running RAP on GSM8K [1]
### RAP
```bash
python -m torch.distributed.run --nproc_per_node 4 examples/rap_gsm8k/inference.py --llama_size "30B" --output_trace_in_each_iter
```
### PAL + Guided Beam Search

> Note: You need to apply for the [research access](https://openai.com/form/researcher-access-program) to `Codex` (`code-davinci-002`) to run this approach
```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/guided_gsm8k/inference.py --n_actions 16 --temperature 1.0 --reward_alpha 0.5 \
    --beam_size 5 --sampling_strategy stochastic --replace False \
    --beam_search_temperature 0.5 --beam_search_temperature_decay 1.0
```


## Blocksworld
### Preparation
1. Download the validator from [here]()

### RAP
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 examples/rap_blocksworld/inference.py --llama_size "30B" --data_path 'examples/rap_blocksworld/data/step_4.json' --depth_limit 4 --output_trace_in_each_iter
```


## References
[1] Cobbe, Karl, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert et al. "Training verifiers to solve math word problems." arXiv preprint arXiv:2110.14168 (2021).
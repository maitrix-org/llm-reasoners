# What's new

For every dataset, we want to have 10 examples. During inference, the in-context demonstrations are supposed to be randomly sampled.

To support that, we will need to set up a new class for a dataset, see `GSM8KEvaluator` in the run below. More common functions (e.g., the loop over dataset) are also moved into this class. That will make our main entry point cleaner.

Another thing to note is that, we will need to update the prompt in `update_example` in SearchConfig and WorldModel. Check the example below.

I have made least-to-most + MCTS working on the new evaluation setting.

TODO:
- Modify `GSM8KEvaluator` to support RAP and CoT prompts
    - RAP prompting: Should be easily induced from my current prompt for L2M.
    - CoT prompting: May need some human efforts to rewrite my current prompt into CoT format. Copilot can should a lot to make this process faster.
- Write a base class for all evaluator.

```bash
CUDA_VISIBLE_DEVICES=2,3 python examples/rap_gsm8k_l2m/inference_new.py \
--base_lm exllama \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map '[16,22]' \
--n_iters 1 \
--depth_limit 5 \
--n_confidence 1 \
--n_action 1
```
# LLM-search

## To-do
- [ ] Update algorithm interface (allows for more return values as a dict)
- [x] Add a language model
- [x] Implement GSM beam search

## Commands
- `python -m torch.distributed.run --nproc_per_node 2 --master-port 1074 examples/example_gsm8k_beamsearch.py --llama_path $LLAMA_CKPTS --llama_size 13B --prompt_path examples/prompts/example_gsm8k_prompt.json`
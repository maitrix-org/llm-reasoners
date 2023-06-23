# LLM-search

## To-do
- [ ] Update algorithm interface (allows for more return values as a dict)
- [x] Add a language model
- [x] Implement GSM beam search

## Commands
- `python -m torch.distributed.run --nproc_per_node 2 --master-port 1074 examples/example_gsm8k_beamsearch.py --llama_path $LLAMA_CKPTS --llama_size 13B --prompt_path examples/prompts/example_gsm8k_prompt.json`



## Results

|Methods|GSM8K|AQuA|SVAMP|ASDiv|CommonsenseQA|StrategyQA|
|-|-|-|-|-|-|-|
|Direct Prompting||
|CoT|
|Least-to-Most|
|Beam Search|
|ToT|
|RAP|
|CoT+SC|
|Least-to-Most+SC|
|Beam Search - aggr|
|ToT - aggr|
|RAP - aggr|



|Methods|Blocksworld|Game of 24|Mini Crosswords|ProntoQA|
|-|-|-|-|-|
|Direct Prompting|
|CoT|
|Least-to-Most|
|Beam Search|
|ToT|
|RAP|


- explain why GSM is slow
- explain the trick on code
- add as many comments as possible
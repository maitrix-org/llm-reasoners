# Information on OSS and closed-source models

## running experiments

Launching experiments w/ OSS LLMs is not fully automatic yet.
1. launch your supported OSS LLM(s) using the `llm_configs.py`
    - set the `model` var in the dunder main to a model in `CHAT_MODEL_ARGS_DICT` 
    - launch `llm_configs.py`
2. retrieve the `url` that the eai cli will output in the terminal
    - run `eai job ls -r` to print it
3. set the `model_url` var of the OSS LLM(s) in `CHAT_MODEL_ARGS_DICT` to the `url` above
4. in `exp_configs_OSS.py`, set `model_name_list` to the OSS LLM(s) you just launched
5. launch your favorite `exp_config` w/in the `exp_config_OSS.py`    


## Supported OSS LLMs


### Deepseek-AI

Solid team delivering Code LLMs

- `deepseek-ai/deepseek-coder-6.7b-instruct`
    - SOTA small-size CodeLLM w/ 16k context window

### CodeLLAMA
Finetuned LLAMA2 on code, then long-context (16k) then instructions

- `codellama/CodeLlama-7b-Instruct-hf`
- `codellama/CodeLlama-13b-Instruct-hf`
- `codellama/CodeLlama-34b-Instruct-hf`
- `codellama/CodeLlama-70b-Instruct-hf` (4k context window)

### Bigcode

- `bigcode/starcoder2`
    - 15b codeLLM base model w/ 4k sliding window --> 16k total context window. 
- `bigcode/starcoder2-7b`
- `HuggingFaceH4/starchat2-15b-v0.1`
    - chat-finetuned starcoder2 **but not instruct-finetuned**
- `bigcode/starcoder`
    - 15b codeLLM base model w/ 8k context window
- `bigcode/Starcoderplus`
    - python then instruct-finetuned.
- `HuggingFaceH4/starchat-beta` is then chat-finetuned.



## Tentative OSS LLMs

### Salesforce

- `Salesforce/xLAM-v0.1-r`
    - Mixtral base finetuned on lots for agent data

### Cohere

- Command R and Command R +
    - 128k context window and good at using it (RAG training)
    - solid at using tools and APIs
    - we can use the RAG to breakdown the large HTMLs


### Databricks

- `databricks/dbrx-instruct`
    - 32k context window
    - pretty strong! 

### MistralAI

MistralAI delivers strong models w/ large context window. 

- `mistralai-community/Mixtral-8x22B-v0.1` (64k)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (32k)
- `mistralai/Mistral-7B-Instruct-v0.2` (32k)

### AgentLM
- `THUDM/agentlm-70b`
    - Finetuned LLAMA2-70b on text2actions
    - mind the 4k context window

### Salesforce's Agents

TODO

### Acknowledge OSS LLMs

- `codegemma`
    - ok but not as good as `deepseekcoder` and half the context window (8k vs 16k)

## Tentative VLMs

- `adept/fuyu-8b`
    - in their demo, they queried the SNOW UI!


## Relevant Benchmarks

- [bigcode/bigcode-models-leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
- [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)


- Maybes:
    - [Tool talk](https://github.com/microsoft/ToolTalk)
    - [mteb/leaderboard]() embedding leaderboard for RAG

## Close source

### GPTs

TODO

### Claudes

TODO

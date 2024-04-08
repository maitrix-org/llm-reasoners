# Examples

## GSM8K
### RAP
- LLaMA
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 6666 examples/rap_gsm8k/inference.py --base_lm llama --llama_ckpt /path/to/llama_ckpts --llama_size 13B
  ```
- Llama2
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 6676 examples/rap_gsm8k/inference.py --base_lm llama-2 --llama_2_ckpts /path/to/llama-2-ckpts --llama_size 13B
  ```
- ExLlama
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python examples/rap_gsm8k/inference.py --base_lm exllama --exllama_model_dir TheBloke/Llama-2-70B-GPTQ --exllama_lora_dir None --exllama_mem_map '[16,22]'
  ```
  
- Hugging Face
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python examples/rap_gsm8k/inference.py --base_lm hf --hf_path meta-llama/Llama-2-70b-hf --hf_peft_path None --hf_quantized 'nf4'
  ```
  > Note: You need to [request access to `meta-llama/Llama-2-70b-hf` from Meta AI](https://github.com/facebookresearch/llama#access-on-hugging-face). Alternatively, You can also try other models on Hugging Face.
- llama.cpp
  ```bash
  CUDA_VISIBLE_DEVICES=0 python examples/rap_gsm8k/inference.py --base_lm llama.cpp --llama_cpp_path /path/to/13B/ggml-model-q5_0.gguf
  ```
### AQuA
ExLlama
```bash
CUDA_VISIBLE_DEVICES=1,5 python examples/AQuA/inference.py
```

### PAL + Guided Beam Search

> Note: You need to apply for the [research access](https://openai.com/form/researcher-access-program) to `Codex` (`code-davinci-002`) to run this approach

#### Beam Search (Stochastic Sampling) 

```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/guided_gsm8k/inference.py --n_actions 16 --temperature 1.0 --reward_alpha 0.5 \
    --beam_size 5 --sampling_strategy stochastic --replace False \
    --reject_sample True --reject_min_reward 0.6 --unbiased True \
    --beam_search_temperature 0.5 --beam_search_temperature_decay 1.0 \
    --majority_voting_n 4 
```

#### Beam Search (Top-k Sampling) 

```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/guided_gsm8k/inference.py --n_actions 16 --temperature 1 --reward_alpha 0.5 \
    --beam_size 5 --sampling_strategy argmax 

```

## Blocksworld
### Preparation
1. Download the test cases in the root directory
    ```bash
    git clone https://github.com/karthikv792/gpt-plan-benchmark`
    cd gpt-plan-benchmark
    git checkout bf00a2196e92422d1000abc37dd050ef8186f2ab
    ```
2. Set up `Val` for validation
   1. Install `Val` following this [link](https://github.com/karthikv792/LLMs-Planning/tree/34e6841f81ca7708f2f8b8241504bfe8a908e40b/planner_tools/VAL). Ideally you can directly use the executable files in this directory, and there is no need to build the tool yourself.
   2. Assign path of the folder to the environment variable VAL `export VAL=/path/to/val`

### RAP

> Note: The commands below might be outdated. Please look at the scripts under `blocksworld` for updated commands.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 examples/rap_blocksworld/inference.py --llama_size "30B" --data_path 'examples/rap_blocksworld/data/step_4.json' --depth_limit 4 --output_trace_in_each_iter
```

for llama2
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node 8 examples/rap_blocksworld/inference.py --llama_size "70B" --data_path 'examples/rap_blocksworld/data/step_4.json' --depth_limit 4 --output_trace_in_each_iter
```

for exllama

```bash
CUDA_VISIBLE_DEVICES=0 python examples/rap_blocksworld/inference.py --data_path 'examples/rap_blocksworld/data/step_4.json' --depth_limit 4 --model_dir 'path/to/model/dir' --lora_dir None --batch_size 1 --output_trace_in_each_iter
```

for huggingface llama

```bash
CUDA_VISIBLE_DEVICES=0 python examples/rap_blockworld/inference.py -data_path 'examples/rap_blocksworld/data/step_4.json' --depth_limit 4 --hf_path 'path/to/hf/model/dir' --peft_path None --batch_size 1 --quantized 'nf4' --output_trace_in_each_iter
```

## StrategyQA
### RAP
user need to switch the main function for Fire()

```bash
python -m torch.distributed.run --nproc_per_node 4 examples/rap_strategyQA/inference.py --llama_size "30B" --interactive_prompt $interactive_prompt --useful_prompt $useful_prompt --decompose_prompt $decompose_prompt --output_trace_in_each_iter
```

## Game of 24
> Note: You need to make a directory and put the game24 data in it. For example, examples/tot_game24/data/24.csv

### ToT (Beam Search)
```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/tot_game24/inference.py --batch_size 2 --model gpt-3.5-turbo --temperature 0.7
```

## Crosswords
> Note: You need to make a directory and put the MiniCrosswords data in it. For example, examples/tot_crosswords/data/mini0505.json

### ToT (DFS on MiniCrosswords)
```bash
export OPENAI_API_KEY=YOUR_API_KEY
python examples/tot_crosswords/inference.py --model gpt-4 --temperature 0.7
```

## References
[1] Cobbe, Karl, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert et al. "Training verifiers to solve math word problems." arXiv preprint arXiv:2110.14168 (2021).

[2] Valmeekam, Karthik, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. "Large Language Models Still Can't Plan (A Benchmark for LLMs on Planning and Reasoning about Change)." arXiv preprint arXiv:2206.10498 (2022).

[3] Yao, Shunyu, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. "Tree of thoughts: Deliberate problem solving with large language models." arXiv preprint arXiv:2305.10601 (2023).

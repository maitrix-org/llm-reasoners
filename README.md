# 🤔Reasoners
Reasoners is a toolkit to support advanced reasoning with LLMs, especially tree-structured reasoning (e.g., [RAP](https://arxiv.org/abs/2305.14992), [ToT](https://arxiv.org/abs/2305.10601), [Guided Decoding](https://arxiv.org/abs/2305.00633), etc.). With Reasoners, it's easy to apply state-of-the-art LLMs (Open-sourced models or OpenAI API) to any problems you want to solve with any reasoning algorithms. The reasoning tree can be visualized with a line of code.

## Why Reasoners?
- **Unified Formulation**: We regard reasoning problems as decision making problems with certian action/state definitions. This formulation covers most popular reasoning algorithms and enable a unified interface for all of them. Users only need to define the state transition and some search configurations to work on a new domain.
- **Visualization**: We provide visualization tools to help users understand the reasoning process. Even for the most complicated reasoning algorithms, e.g. Monte-Carlo Tree Search, users can easily diagnose what happened.
- **LLaMA Integration**: We integrate the state-of-the-art open-sourced LLM: LLaMA, and implement many helper functions, making it as versatile as HuggingFace while as fast as the official implementation.


## Online Demo
> TBA

## Quick Tour
> We need a very short example here, perhaps game of 24?

## Installation
```bash
git clone https://github.com/Ber666/Reasoners
cd Reasoners
pip install .
```
Note some optional modules (e.g. local visualization) may need other dependencies. Please refer to the error message for details.

## Benchmarks
We tested different reasoning algorithms on first 100 examples of the following benchmarks (To be updated). Superscripted rows indicate the reported results in the original paper.

|Methods|Base LLM|GSM8K|AQuA|SVAMP|ASDiv|CommonsenseQA|StrategyQA|
|-|-|-|-|-|-|-|-|
|CoT|-|-|-|-|-|-|-|
|CoT+SC|-|-|-|-|-|-|-|
|Least-to-Most+SC|-|-|-|-|-|-|-|
|Guided Decoding<sup>[[1]](https://arxiv.org/abs/2305.00633)</sup>|CodeX (PAL)|-|-|-|-|-|-|
|Guided Decoding|LLaMA-65B (PAL)|-|-|-|-|-|-|
|RAP - BeamSearch|-|-|-|-|-|-|-|
|RAP - MCTS|-|-|-|-|-|-|-|
|RAP - MCTS - aggr|


|Methods|Base LLM|Blocksworld|Game of 24|Mini Crosswords|ProntoQA|
|-|-|-|-|-|-|
|CoT|-|-|-|-|-|
|ToT<sup>[[2]](https://arxiv.org/abs/2305.10601)<sup>|-|-|-|-|-|
|ToT|-|-|-|-|-|
|RAP|LLaMA-33B|-|-|-|-|

## Using `llama.cpp`
We provide the wrapper for `llama-cpp-python`, which can handle 8-bit or even 4-bit quantized version of LLaMA.
For how to quantize LLaMA, refer to [llama.cpp](https://github.com/ggerganov/llama.cpp).
To install `llama-cpp-python` with CUDA acceleration, run
```bash
LLAMA_CUBLAS=1 CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose
```
Test the installation by running `python -m llama_cpp`.
If successful with CUDA acceleration, you should get `ggml_init_cublas: found <num> CUDA devices`.
It is normal if there is a following cuBLAS error.

If there is an error about `nvcc not found`, you may try in a new conda environment
```bash
conda install -c "nvidia/label/cuda-12.0.0" libcublas cuda-toolkit
```
replacing `12.0.0` with your cuda version.
For more details and troubleshooting, please refer to [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

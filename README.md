![logo](images/image.png#pic_center)


---


**LLM Reasoners** is a library to support advanced reasoning with LLMs. We formulate multi-step reasoning as decision-making, where each reasoning step is an action. A user could define the problem they want to work on, and LLM reasoners would provide you with anything else (Search Algorithms, Visualization, LLM calling, etc.)!

## Why Reasoners?
- **Unified Formulation**: A Reasoner is composed of a `SearchConfig` to formulate the action space and reward, with an optional `WorldModel` to customize the state transition. This minimizes the workload to reason on new problems with LLMs, but also supports diverse types of reasoning problems, ranging from Question Answering to Embodied Plan Generation.
- **Latest Algorithms**: We provide the latest search algorithms for reasoning, including [RAP](https://arxiv.org/abs/2305.14992), [ToT](https://arxiv.org/abs/2305.10601), [Guided Decoding](https://arxiv.org/abs/2305.00633), etc. These algorithms enable tree-structure reasoning and are essentially superior to chain-of-thoughts.
- **Visualization**: Visualization tools are available to help users understand the reasoning process. Users can easily diagnose what happened even for the most complicated reasoning algorithms, e.g., Monte-Carlo Tree Search.
- **LLM Wrapper**: Our framework is compatible with any LLM framework, and we specifically wrap LLaMA with some common helper functions to make it easier to use. We support LLaMA with [fairscale](https://github.com/facebookresearch/llama) backend for better multi-GPU performance, or [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) backend with less hardware requirement. 

## Online Demo
> TBA

## Quick Tour
> TBA

## Installation
```bash
git clone https://github.com/Ber666/llm-reasoners
pip install .
```
Note that some optional modules may need other dependencies. Please refer to the error message for details.

## Benchmarks
We tested different reasoning algorithms on first 100 examples of the following benchmarks (To be updated). Superscripted rows indicate the reported results in the original paper.

|Methods|Base LLM|GSM8K|AQuA|SVAMP|ASDiv|CommonsenseQA|StrategyQA|
|-|-|-|-|-|-|-|-|
|CoT|-|-|-|-|-|-|-|
|CoT+SC|-|-|-|-|-|-|-|
|Least-to-Most+SC|-|-|-|-|-|-|-|
|Guided Decoding<sup>[[1]](https://arxiv.org/abs/2305.00633)</sup>|CodeX (PAL)|-|-|-|-|-|-|
|Guided Decoding|CodeX (PAL)|-|-|-|-|-|-|
|RAP - BeamSearch|-|-|-|-|-|-|-|
|RAP - MCTS|-|-|-|-|-|-|-|
|RAP - MCTS - aggr|-|-|-|-|-|-|-|


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

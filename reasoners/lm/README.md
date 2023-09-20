# Language Models
## LLaMA
- Please follow MetaAI's [instruction](https://github.com/facebookresearch/llama) to get the models and set up the environment.
- Example run:
  - LLaMA
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 6666 examples/rap_gsm8k/inference.py --base_lm llama --llama_ckpt /path/to/llama_ckpts --llama_size 13B
    ```
  - Llama 2
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 6676 examples/rap_gsm8k/inference.py --base_lm llama-2 --llama_2_ckpts /path/to/llama-2-ckpts --llama_size 13B
    ```
  - If you set `export LLAMA_CKPTS=/path/to/llama_ckpt` and/or `export LLAMA_2_CKPTS=/path/to/llama_2_ckpt`, you can omit `--llama_ckpts` and/or `--llama_2_ckpts`.
  - `--nproc_per_node` depends on the model size, please check the [instruction](https://github.com/facebookresearch/llama).

## ExLlama
- We provide support for [ExLlama](https://github.com/turboderp/exllama) as a submodule.
- Choose any quantized [models](https://github.com/turboderp/exllama/blob/master/doc/model_compatibility.md) available from Hugging Face. You can also use the path of a local quantized model in the same format as Hugging Face.
- Example run: Llama-2 70B on **2 * 24GB GPUs**:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python examples/rap_gsm8k/inference.py --base_lm exllama --exllama_model_dir TheBloke/Llama-2-70B-GPTQ --exllama_lora_dir None --exllama_mem_map '[16,22]'
  ```
- If you clone our repo without `--recursive`, you can run `git submodule update --init` for ExLlama submodule to work.

## Hugging Face
- We provide a wrapper for Hugging Face models
- We support `8bit`, `nf4`, `fp4`, or `awq` as optional quantization method
  - Additional dependencies are requried to use `awq`, which can be installed by running `pip install -e '.[awq]'` under the root of this repository
- Example run:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python examples/rap_gsm8k/inference.py --base_lm hf --hf_path meta-llama/Llama-2-70b-hf --hf_peft_path None --hf_quantized 'nf4'
  ```

## llama.cpp
- We provide the wrapper for `llama-cpp-python`.
For how to quantize LLaMA, refer to [llama.cpp](https://github.com/ggerganov/llama.cpp).
- To install `llama-cpp-python` with CUDA acceleration, run
    ```bash
    LLAMA_CUBLAS=1 CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose
    ```
- Test the installation by running `python -m llama_cpp`.
  - If successful with CUDA acceleration, you should get `ggml_init_cublas: found <num> CUDA devices`.
  - It is usually fine if there is a cuBLAS error shown.
  - If there is an error about `nvcc not found`, you may try in a new conda environment (replacing `12.0.0` with your cuda version):
    ```bash
    conda install -c "nvidia/label/cuda-12.0.0" libcublas cuda-toolkit
    ```
- For more details and troubleshooting, please refer to [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Example run:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python examples/rap_gsm8k/inference.py --base_lm llama.cpp --llama_cpp_path /path/to/13B/ggml-model-q5_0.gguf
  ```
- From our experiments, `llama.cpp` suffers from slow inference speed when the model is put on multiple GPUs. Please contact us if you know how to fix this issue.

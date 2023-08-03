# Language Models
## LLaMA
- Please follow MetaAI's [instruction](https://github.com/facebookresearch/llama/tree/llama_v1) to get the models and set up the environment.
- If you want to run our examples with LLaMA, please set the path to checkpoint with `export LLAMA_CKPTS=/path/to/llama/checkpoints`.
- for llama2 `export LLAMA_2_CKPTS=/path/to/llama/checkpoints`
- Note that this version of LLaMA is based on `fairscale`, and you may use `python -m torch.distributed.run --n_proc_per_node num_proc your_script.py` to run the script (`num_proc` depends on the model size, please check the [instruction](https://github.com/facebookresearch/llama/tree/llama_v1)).
- We are working on the integration of LLaMA-2 into our framework. Stay tuned!

## llama.cpp
- We provide the wrapper for `llama-cpp-python`, which can handle 8-bit or even 4-bit quantized version of LLaMA.
For how to quantize LLaMA, refer to [llama.cpp](https://github.com/ggerganov/llama.cpp).
- To install `llama-cpp-python` with CUDA acceleration, run
    ```bash
    LLAMA_CUBLAS=1 CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --verbose
    ```
- Test the installation by running `python -m llama_cpp`.
  - If successful with CUDA acceleration, you should get `ggml_init_cublas: found <num> CUDA devices`.
  - It is usually fine if there is a cuBLAS error shown.
  - If there is an error about `nvcc not found`, you may try in a new conda environment(replacing `12.0.0` with your cuda version):
    ```bash
    conda install -c "nvidia/label/cuda-12.0.0" libcublas cuda-toolkit
    ```
- For more details and troubleshooting, please refer to [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [llama.cpp](https://github.com/ggerganov/llama.cpp).
- From our experiments, `llama.cpp` suffers from slow inference speed when the model is put on multiple GPUs. Please contact us if you know how to fix this issue.
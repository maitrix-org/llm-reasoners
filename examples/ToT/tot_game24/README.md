## Run
An example for llama2
```bash
CUDA_VISIBLE_DEVICES=0 python examples/tot/tot_game24/inference.py --base_lm llama-2 --llama_2_ckpts your/path/to/llama --llama_size "7B"   --batch_size 8 --n_iters 1
```
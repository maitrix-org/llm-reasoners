# Grace Decoding

## Setup
Please clone the [GRACE](https://github.com/mukhal/grace.git) repo and copy the modified `transformers` library from GRACE repo into the llm-reasoners grace_gsm8 directory. Rename the `transformers` directory to `transformers_grace`. Install the `transformers_grace` package by running the following commands:
  ```
  cd examples/grace/grace_gsm8k/transformers_grace/
  pip install -e .
  ```
## Prepare the discriminator model through [GRACE](https://github.com/mukhal/grace.git)
After cloning the [GRACE](https://github.com/mukhal/grace.git) repo, please `cd grace` and run:
  ```
pip install huggingface_hub
python download_models.py --task gsm8k
  ```
Then discriminator_model is saved to  `grace/ckpts/discrim/'


##  RUN
Use Flan T5 to run
  ```bash
  CUDA_VISIBLE_DEVICES=0,1  python examples/Grace/gsm8k/inference.py --base_lm mkhalifa/flan-t5-large-gsm8k --discriminator_path your/path/to/discriminator_model
  ```

Use llama3 to run
```bash
  CUDA_VISIBLE_DEVICES=0  torchrun --nproc-per-node 1 --master-port 1234  examples/Grace/gsm8k/inference.py --base_lm llama-3 --discriminator_path your/path/to/discriminator_model --llama_3_ckpts $LLAMA3_CKPTS --llama_size "8B-Instruct" 
  ```

P.S. If an error occurs, you can solve the problem according to the error message 
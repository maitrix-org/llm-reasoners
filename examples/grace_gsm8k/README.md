# Examples

## GSM8K
### Grace Decoding

#### Setup
Please clone the [GRACE](https://github.com/mukhal/grace.git) repo and copy the modified `transformers` library from GRACE repo into the llm-reasoners grace_gsm8 directory. Rename the `transformers` directory to `transformers_grace`. Install the `transformers_grace` package by running the following commands:
  ```
  cp -r path/to/grace/transformers examples/grace_gsm8k/
  cd examples/grace_gsm8k/transformers_grace/
  pip install -e .
  ```
#### Flan T5
  ```bash
  CUDA_VISIBLE_DEVICES=0,1  python examples/grace_gsm8k/inference.py --base_lm mkhalifa/flan-t5-large-gsm8k
  ```
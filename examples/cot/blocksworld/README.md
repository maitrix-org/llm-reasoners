# Blocksworld

## Preparation

1. Download the test cases in the root directory  (`llm-reasoners/`)

   ```bash
   git clone https://github.com/karthikv792/LLMs-Planning.git
   ```

2. Set up `Val` for validation

   1. Install `Val` following this [link](https://github.com/karthikv792/LLMs-Planning/tree/34e6841f81ca7708f2f8b8241504bfe8a908e40b/planner_tools/VAL). Ideally you can directly use the executable files in this directory, and there is no need to build the tool yourself.

   ```bash
    cd LLMs-Planning/planner_tools/VAL
    make validate
    make parser
   ```

   2. Assign path of the folder to the environment variable VAL `export VAL=/path/to/val`

## Data Description

The full [Blocksworld](https://arxiv.org/abs/2305.15771) datasets contain 602 samples.

Our experiments are conducted in two distinct settings: Hard (v1) and Easy (v2). In Easy setting, we assume prior knowledge of the minimum number of actions for each case. Leveraging this information, we use demonstration cases that share the same minimum number of actions as the test case. For each group of cases, we randomly select 10 cases to create a pool of demonstration cases, leaving the remaining cases as the test set. During inference, we randomly sample 4-shot demonstration cases from this pool and utilize them to
formulate prompts. In the Hard setting, we randomly select 10 cases from the full dataset to form a demonstration pool and subsequently exclude these cases from the test set. During inference, we randomly sample 4-shot demonstration cases from this global pool, irrespective of the minimum number of actions required for the test case.


## Run

We provide the script to reproduce the results of COT (Hard one) 

To reproduce the result, make sure:

1. you have prepared the corresponding LLM(e.g. `Llama-2-13b-hf` or `Llama2-7B`) 
2. Modify the `CUDA_VISIBLE_DEVICES`, `model_path` and `llama_size`(if have) in the shell you will run. (e.g. `test_cot_llama13b.sh` or  `test_cot_llama2.sh`) 

Then run the corresponding shell:
```bash
./examples/cot/blocksworld/test_cot_llama2.sh
```

If you want to modify the experiment settings or develop your own method, you may look at `cot_inference.py`.
## How to use RICE

Here I will show you how to use the RICE to evaluate your own Chain-of-Thought result or datasets in our paper.

#### Set the OpenAI_Key

```
export OPENAI_API_KEY= YOUR_OWN_OPEN_AI_KEY
```

#### Reproduce the evaluation in paper

```
cd .../RICE
python rice.py --dataset DATASET_NAME --prompt_type PROMPT_TYPE
```

The `DATASET_NAME` is one of `['gsm8k','strategyqa','AQuA','cosmos', 'multistep_arithmetic','word_sorting','logical_deduction']` and the `PROMPT_TYPE` can be found in `prompt.json`

#### Evaluate your own result

Theorically, RICE can support any evaluation of Chain-of-Thought. In practice, you could first use the `RICE_criterion()` function in `rice.py` to generate your own dataset's criterion prompt first and take `RICE_evaluation()` for evaluation. We showed an example from AQuA.
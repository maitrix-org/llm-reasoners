# Blocksworld

## Data
The full [Blocksworld](https://arxiv.org/abs/2305.15771) datasets contain 602 samples.

Our experiments are conducted in two distinct settings: Hard (v1) and Easy (v2). In Easy setting, we assume prior knowledge of the minimum number of actions for each case. Leveraging this information, we use demonstration cases that share the same minimum number of actions as the test case. For each group of cases, we randomly select 10 cases to create a pool of demonstration cases, leaving the remaining cases as the test set. During inference, we randomly sample 4-shot demonstration cases from this pool and utilize them to
formulate prompts. In the Hard setting, we randomly select 10 cases from the full dataset to form a demonstration pool and subsequently exclude these cases from the test set. During inference, we randomly sample 4-shot demonstration cases from this global pool, irrespective of the minimum number of actions required for the test case.

We provide the script to reproduce the results of [CoT](test_cot.sh) (for both easy and hard) and RAP ([easy](test_rap_v2.sh) and [hard](test_rap_v1.sh)).

If you want to modify the experiment settings or develop your own method, you may look at `cot_inference.py` or `rap_inference.py`.

The [data files](data) may contain absolute paths, please replace them with your own paths before running.
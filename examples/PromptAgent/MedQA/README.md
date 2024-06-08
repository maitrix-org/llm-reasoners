# Prompt Agent

This example demonstrates the implementation of the Prompt Agent framework ([Wang et al., 2023](https://arxiv.org/abs/2310.16427)) using the LLM Reasoners. The goal is to enhance task prompts to an expert level through strategic planning and error feedback.


## Introduction

PromptAgent optimizes prompts by treating the process as a strategic planning problem. Explore various potential prompts, iterating through states (prompt versions) and actions (modifications based on model errors) to systematically refine them. This method bridges the gap between novice and expert prompt engineers with minimal human intervention, ensuring maximum performance.

## Running the example

Prerequisites:
- Set the environment variable `OPENAI_API_KEY` to your API key. Each run may cost approximately $5 USD.

How to run:
- python inference.py --prompt "The origin prompt"

## Results

We tested the performance using a subset of the MedQA data set, provided in the data.json file. The training dataset consists of 2000 questions, the evaluation  dataset contains 150 questions, the test dataset contains 5000 questions. The hyperparameter settings are the same as those used in the PromptAgent Lite configuration.

### Original Prompt
Please use your domain knowledge in medical area to solve the questions.

### Optimized Prompt
Please draw upon your comprehensive medical knowledge to thoroughly analyze the questions provided. Consider all aspects of the patient's history, lifestyle choices, and potential risk factors in your responses. Reflect on the importance of differential diagnosis and the integration of symptoms with clinical findings to arrive at the most accurate medical conclusions. Remember to include how the patient's medication use and lifestyle can impact diagnostic outcomes. Ensure that your analysis is aligned with the latest medical standards and guidelines, and clearly indicate your final answer. Let's aim for a holistic and precise diagnosis based on the patient's complete scenario.

|Promp|Eval Accuracy|Test Accuracy
|-|-|
|Original Prompt|0.367|0.374
|Optimized Prompt|0.5066666666666667|0.524
 

## Reference
```bibtex
@article{wang2023promptagent,
  title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
  author={Wang, Xinyuan and Li, Chenxi and Wang, Zhen and Bai, Fan and Luo, Haotian and Zhang, Jiayou and Jojic, Nebojsa and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2310.16427},
  year={2023}
}
```

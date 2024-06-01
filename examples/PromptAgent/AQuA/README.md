# Prompt Agent

This example demonstrates the implementation of the Prompt Agent framework ([Wang et al., 2023](https://arxiv.org/abs/2310.16427)) using the LLM Reasoners. The goal is to enhance task prompts to an expert level through strategic planning and error feedback.


## Introduction

PromptAgent optimizes prompts by treating the process as a strategic planning problem. Explore various potential prompts, iterating through states (prompt versions) and actions (modifications based on model errors) to systematically refine them. This method bridges the gap between novice and expert prompt engineers with minimal human intervention, ensuring maximum performance.

## Running the example

Prerequisites:
- Set the environment variable `OPENAI_API_KEY` to your API key. Each run may cost approximately $5 USD.

How to run:
- python inference.py "The origin prompt"

## Results

We tested the performance using a subset of the AQuA data set, provided in the data.json file. The training dataset consists of 180 questions, while the evaluation test set contains 70 questions. The hyperparameter settings are the same as those used in the PromptAgent Lite configuration.

|Prompt|Accuracy|
|-|-|
|Original Prompt|0.543|
|Optimized Prompt| 0.671|
 

## Reference
```bibtex
@article{wang2023promptagent,
  title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
  author={Wang, Xinyuan and Li, Chenxi and Wang, Zhen and Bai, Fan and Luo, Haotian and Zhang, Jiayou and Jojic, Nebojsa and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2310.16427},
  year={2023}
}
```

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
Draw upon your comprehensive medical expertise and critical thinking capabilities to provide a precise answer for the forthcoming query. In formulating your response, foreground your rationale in up-to-date clinical guidelines, deep physiological understanding, or pharmacological insights where relevant. Your explanation should not only demonstrate a mastery of the medical context but also engage in logical deduction rooted in the most relevant medical evidence available. Clearly articulate your selected answer, delving into a stepwise explanation that underscores how it coheres with contemporary medical practices and understandings. Additionally, pay special attention to integrating all aspects of the clinical presentation, including patient history, symptoms, examination findings, and laboratory data, to support your diagnostic or therapeutic conclusion. Ensure that your analysis embodies a rigorous application of medical knowledge, effectively synthesizing information to align with the principles of evidence-based medicine.


|Promp|Eval Accuracy|Test Accuracy
|-|-|-|
|Original Prompt|0.413|0.464
|Optimized Prompt|0.587|0.572
 

## Reference
```bibtex
@article{wang2023promptagent,
  title={PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization},
  author={Wang, Xinyuan and Li, Chenxi and Wang, Zhen and Bai, Fan and Luo, Haotian and Zhang, Jiayou and Jojic, Nebojsa and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2310.16427},
  year={2023}
}
```

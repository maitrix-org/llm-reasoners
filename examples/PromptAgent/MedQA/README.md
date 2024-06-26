# Prompt Agent

This example demonstrates the implementation of the Prompt Agent framework ([Wang et al., 2023](https://arxiv.org/abs/2310.16427)) using the LLM Reasoners. The goal is to enhance task prompts to an expert level through strategic planning and error feedback.


## Introduction

PromptAgent optimizes prompts by treating the process as a strategic planning problem. Explore various potential prompts, iterating through states (prompt versions) and actions (modifications based on model errors) to systematically refine them. This method bridges the gap between novice and expert prompt engineers with minimal human intervention, ensuring maximum performance.

## How to make custom tasks

### config.py
#### Hyperparameters
- `depth_limit`: The maximum depth for searching.
- `origin_prompt`: Initial prompt.
- `num_batches`: Number of batches.
- `steps_per_gradient`: Number of prompts per gradient computation.
- `batch_size`: Number of items per batch.
- `w_exp`: Exponential weight factor for computations.
- `n_iters`: Number of iterations for each reasoning process.
- `prompt_position`: Positioning of the prompt in the reasoning process, set to "pre" to indicate before the main content, 'pro' to then end.

#### Model Configuration
- `base_model`: The model used to answer the questions.
- `optimize_model`: The model used to give the feedback and the new prompts.

### task.py

#### `load_task_dataset`
- **Usage**: Loads the question dataset and preprocesses it into training, evaluating, and testing subsets.
- **Parameters**: None
- **Returns**: Three lists containing questions for training, evaluating, and testing.

#### `reformat_data`
- **Usage**: Reformats the dataset entries for compatibility with the prompt generation process.
- **Parameters**: `question_list` - a list of question entries to be reformatted.
- **Returns**: A reformatted list of question entries. Include keys 'question' and 'answer'

#### `extract_answer`
- **Usage**: Extracts the answer from a response provided by a language model.
- **Parameters**: `message` - the string containing the model's response.
- **Returns**: The extracted answer.

#### `check_answer`
- **Usage**: Checks if the answer provided by the model matches the correct answer.
- **Parameters**:
  - `model_answer`: The answer provided by the model.
  - `correct_answer`: The correct answer to the question.
- **Returns**: `True` if the answers match, otherwise `False`.

## Running the example

Prerequisites:
- Set the environment variable `OPENAI_API_KEY` to your API key. Each run may cost approximately $5 USD.

How to run:
- python inference.py


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

Contributor: [Enze Ma](https://github.com/sora1998) [[LinkedIn](https://www.linkedin.com/in/enze-ma-a9a20a215)] [[Twitter](https://x.com/MaEnze98259)]

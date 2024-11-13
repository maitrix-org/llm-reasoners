# Dynamic Rewarding with Prompt Optimization (DRPO)

This part is the implementation of **Dynamic Rewarding with Prompt Optimization (DRPO)** in LLM Reasoner, as introduced by [Somanshu et al., 2024](https://aclanthology.org/2024.emnlp-main.1220/). DRPO is the first tuning-free, inference-time algorithm designed to self-align large language models (LLMs) with human preferences.

## Introduction

**Dynamic Rewarding with Prompt Optimization (DRPO)** is a novel approach for aligning LLMs without requiring extensive tuning or human supervision. By dynamically optimizing prompts based on model feedback, DRPO enhances alignment performance, enabling LLMs to self-correct and adapt effectively to various challenges. This method significantly reduces alignment costs and improves the versatility of LLM applications.

## Key Parameters

### Model Configuration

- **`base_model_name`** (str): Name or path of the base model to be used.
- **`base_model_family`** (str): Family name of the base model (e.g., 'mistral').
- **`eval_model_name`** (str): Model name for evaluation (e.g., 'gpt-4-0125-preview').
- **`metrics_model_name`** (str): Model name for dynamic reward selection.
- **`optimize_model_name`** (str): Model name for optimization tasks.
- **`initial_system_prompt`** (str): Initial system prompt for the model.

### Search Algorithm Configuration

- **`n_actions`** (int): Number of actions sampled in beam search.
- **`temperature`** (float): Controls randomness in model predictions.
- **`depth`** (int): Initial search depth for exploration.
- **`beam_size`** (int): Number of beams for beam search.

### Additional Parameters

- **`max_depth_increase`** (int): Maximum search depth increment, useful for handling low-difficulty training samples.
- **`log_dir`** (Optional[str]): Directory path for storing logs.
- **`disable_log`** (bool): Disables logging if set to `True`.
- **`disable_tqdm`** (bool): Disables tqdm progress bars if set to `True`.
- **`base_model_download_dir`** (str): Directory for downloading base model files.
- **`data_dir`** (str): Directory path for data files.
- **`metrics_cache_path`** (str): Path to cache evaluation metrics.
- **`num_training_examples`** (int): Number of training examples.
- **`logging_level`** (str): Logging level (e.g., "INFO" or "DEBUG").
- **`ret_icl`** (bool): If `True`, optimizes prompt with retrieval-based in-context learning (ICL).
- **`is_GPT`** (bool): Treats the model as GPT if set to `True`.
- **`k`** (int): Number of retrievals.
- **`cuda_visible_devices`** (str): Specifies which CUDA devices are visible.
- **`num_gpus`** (int): Number of GPUs to use.

## Training Outputs

Upon completing training, the following files are generated in the `log_dir`:

1. **`args.txt`**: Stores all specified training arguments.
2. **`log.txt`**: Training log, including model responses and generated rewards.
3. **`algo_output/output.pkl`**: Complete output showing prompts and rewards at each optimization stage.
4. **`algo_output/trace.txt`**: Trace of prompt evolution across the search process.

## Running the Training Process

Start training with a simple command:
Replace <parameters> with any desired arguments to customize your training.

```bash
python inference.py <parameters>
```


## Result

**Initial System Prompt**  
"You are a helpful assistant."

**Improved System Prompt**  
"You are a highly capable assistant designed to provide accurate, ethical, insightful, and creative assistance across a broad spectrum of queries. Your mission is to deliver information and advice that is not only correct but also rich in detail, uniquely tailored to each user's specific context, needs, and emotions. In every interaction, strive to balance factual accuracy with empathy, understanding, a commitment to ethical standards, and a dash of creativity. Illuminate subjects with precision, creativity, and depth, making complex information accessible and engaging. Engage users with a conversational tone, prioritizing clarity, empathy, creativity, and personalization to foster a deeper understanding of the subject matter.  
- You do not have access to the internet or real-time data, and you cannot perform physical actions. Avoid providing advice that could lead to unsafe practices or the use of incorrect tools.  
- Ensure factual accuracy to the best of your ability, and be transparent about the speculative nature of certain responses or when information cannot be verified. If a query involves tasks that require specialized knowledge or tools, emphasize the importance of consulting with experts or using the correct equipment.  
- Strive for creativity in your responses, exploring unique insights, innovative strategies, and multiple perspectives that could offer users fresh perspectives or solutions. Use analogies, storytelling elements, and vivid descriptions to make your responses as engaging, accessible, and as imaginative as possible.  
- Tailor your responses to directly address the user's specific request, avoiding redundancy and ensuring relevance. Focus on providing alternative approaches or additional insights, especially if the user has already attempted a solution.  
- Acknowledge the limitations of your assistance, especially when dealing with speculative or hypothetical scenarios. Offer a starting point for further exploration rather than a definitive solution, and encourage a deeper understanding of the subject matter.  
- Prioritize personalization in your responses, aiming to understand and address the user's specific context, needs, and emotions, making each interaction feel uniquely valuable."

You can see the full trace in the log folder.
## Reference

If you are interested in this method, please also check the official GitHub repository: [https://github.com/Singla17/dynamic-alignment-optimization/tree/master](https://github.com/Singla17/dynamic-alignment-optimization/tree/master)

```bibtex
@inproceedings{singla-etal-2024-dynamic,
    title = "Dynamic Rewarding with Prompt Optimization Enables Tuning-free Self-Alignment of Language Models",
    author = "Singla, Somanshu  and
      Wang, Zhen  and
      Liu, Tianyang  and
      Ashfaq, Abdullah  and
      Hu, Zhiting  and
      Xing, Eric P.",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1220",
    pages = "21889--21909",
    abstract = "Aligning Large Language Models (LLMs) traditionally relies on complex and costly training processes like supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). To address the challenge of achieving alignment without these extensive tuning costs and expensive annotations, we present a novel, tuning-free approach for self-alignment called Dynamic Rewarding with Prompt Optimization (DRPO). Our approach enables self-alignment through a search-based prompt optimization framework, allowing the model to self-improve and generate optimized prompts without additional training or human supervision. The core of DRPO leverages a dynamic rewarding mechanism to identify and rectify model-specific alignment weaknesses, enabling LLMs to adapt quickly to various alignment challenges. Empirical evaluations on eight recent LLMs, including both open- and closed-source, reveal that DRPO significantly enhances alignment performance, enabling base models to outperform their SFT/RLHF-tuned counterparts. Moreover, DRPO's automatically optimized prompts surpass those curated by human experts, demonstrating its superior alignment capabilities. Our findings envision a highly cost-effective and adaptable solution for future alignment research to be further explored.",
}
```

## Contributor

Contributor: [Enze Ma](https://github.com/sora1998) [[LinkedIn](https://www.linkedin.com/in/enze-ma-a9a20a215)] [[Twitter](https://x.com/MaEnze98259)]

![logo](images/image.png#pic_center)

---


**LLM Reasonsers** is a library to enable LLMs to conduct complex reasoning, with advanced reasoning algorithms. It approaches multi-step reasoning as planning and searches for the optimal reasoning chain, and achieves the best balance of exploration vs exploitation with the idea of "World Model" and "Reward".

Given any reasoning problem, simply define the reward function and an optional world model (explained below), and let LLM reasoners take care of the rest, including Search Algorithms, Visualization, LLM calling, and more!


## Why Choose LLM Reasoners?

- **Cutting-Edge Algorithms**: We offer the most up-to-date search algorithms for reasoning with LLMs, such as [RAP-MCTS](https://arxiv.org/abs/2305.14992), [Tree-of-Thoughts](https://arxiv.org/abs/2305.10601), [Guided Decoding](https://arxiv.org/abs/2305.00633), and more. These advanced algorithms enable tree-structure reasoning and outperform traditional chain-of-thoughts approaches.

- **Intuitive Visualization and Interpretation**: Our library provides visualization tools to aid users in comprehending the reasoning process. Even for the most complex reasoning algorithms like Monte-Carlo Tree Search, users can easily diagnose and understand what occurred with one line of python code.

- **Compatibility with LLM libraries**: Our framework is compatible with any LLM framework, e.g. Huggingface transformers, OpenAI API, etc. Specifically, we integrated LLaMA with the option of using [fairscale](https://github.com/facebookresearch/llama) backend for improved multi-GPU performance or [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) backend with lower hardware requirements.

## Understanding LLM Reasoners

Consider the following problem:

![Alt text](images/goal.png)

Let's start with a naive method for LLM reasoning: Prompted with a few examples of problem-solving step by step, an LLM can generate a chain of thoughts (or a sequence of actions) to solve a new problem. For the problem above, the prompt inputted to the LLM and the expected output (in bold) is shown below:


<pre>
I am playing with a set of blocks where I need to arrange the blocks into stacks.

<i>(Example problems and solutions * 4)</i>

[STATEMENT] 
As initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, the red block is on the yellow block, the yellow block is on the table, the blue block is on the table and the orange block is on the table. My goal is to have that the orange block is on top of the blue block and the yellow block on top of the orange block.

[PLAN]
<b>pick up the orange block</b>
<b>stack the orange block on top of the blue block</b>
<b>unstack the red block from on top of the yellow block</b>
<b>put the red block on the table</b>
<b>pick up the yellow block</b>
<b>stack the yellow block on top of the orange block</b>
</pre>


Regarding each reasoning step as an action, we have $a_1=$"*pick up the orange block*", $a_2=$"*stack the orange block on top of the blue block*", and so on. At each time step, the next action is sampled from the LLM conditioned on the previous actions. This simple method is often referred to as [Chain-of-thoughts](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) reasoning. Unfortunately, it doesn't always work for complex reasoning problems. For [Blocksworld dataset](https://arxiv.org/abs/2305.15771) where the problem above comes from, even the strongest GPT-4 model can only reach the success rate of ~30%.

LLM Reasoners formulates the reasoning as planning. Different from Chain-of-thoughts reasoning which autoregressively samples the next action, our goal is to **efficiently search in the reasoning space for the optimal reasoning chain**. To achieve this, two components need to be defined: a world model and a reward function.

- World model defines the state transition, formally $P(s_{i+1} | s_i, a_i)$. A default world model regards the partial solution as the state and simply appends a new action/thought to the state as the transition (as the formulation of [Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)). However, you’ll have the option to design a better world model which predicts and keeps track of a more meaningful state (e.g., environment status, intermediate variable values, etc. Check [RAP](https://arxiv.org/abs/2305.14992) for more examples), thus enhancing the reasoning. For the example shown above, we can naturally define the state as the condition of blocks (e.g., the red block is on the yellow block...), and a world model is to predict the condition of blocks after every potential action.  

- Reward function provides a criterion to evaluate a reasoning step. Ideally, a reasoning chain with a higher accumulated reward should be more likely to be correct. For the example shown above, we can reward actions based on the increased number of accomplished subgoals they lead to. Besides, the likelihood of LLMs generating the action can also be used as a reward, to give the search a good prior.


After we have the world model and reward function, it's time to apply an algorithm to search for the optimal reasoning trace. Here, we show the process of Monte-Carlo Tree Search:

![Alt text](images/mcts_animation.gif)

## Quick Tour
> TBA

## Installation
```bash
git clone https://github.com/Ber666/llm-reasoners
cd llm-reasoners
pip install -e .
```
Note that some optional modules may need other dependencies. Please refer to the error message for details.

## Benchmarks
We tested different reasoning algorithms on first 100 examples of the following benchmarks (to be updated). Superscripted rows indicate the reported results in the original paper.

|Methods|Base LLM|GSM8K|AQuA|SVAMP|ASDiv|CommonsenseQA|StrategyQA|
|-|-|-|-|-|-|-|-|
|CoT|-|-|-|-|-|-|-|
|CoT+SC|-|-|-|-|-|-|-|
|Least-to-Most+SC|-|-|-|-|-|-|-|
|Guided Decoding<sup>[[1]](https://arxiv.org/abs/2305.00633)</sup>|CodeX (PAL)|-|-|-|-|-|-|
|Guided Decoding|CodeX (PAL)|-|-|-|-|-|-|
|RAP - BeamSearch|-|-|-|-|-|-|-|
|RAP - MCTS|-|-|-|-|-|-|-|
|RAP - MCTS - aggr|-|-|-|-|-|-|-|


|Methods|Base LLM|Blocksworld|Game of 24|Mini Crosswords|ProntoQA|
|-|-|-|-|-|-|
|CoT|-|-|-|-|-|
|ToT<sup>[[2]](https://arxiv.org/abs/2305.10601)<sup>|-|-|-|-|-|
|ToT|-|-|-|-|-|
|RAP|LLaMA-33B|-|-|-|-|

## Citation
This project is an extension of the following paper:
```bibtex
@article{hao2023reasoning,
  title={Reasoning with language model is planning with world model},
  author={Hao, Shibo and Gu, Yi and Ma, Haodi and Hong, Joshua Jiahua and Wang, Zhen and Wang, Daisy Zhe and Hu, Zhiting},
  journal={arXiv preprint arXiv:2305.14992},
  year={2023}
}
```

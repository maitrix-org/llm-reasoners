# Examples of using LLM Reasoners
LLM Reasoners now provides the example code for the following methods:
  - [Reasoning-via-Planning, MCTS](RAP) ([Hao et al., 2023](https://arxiv.org/abs/2305.14992))
  - [StructChem](StructChem) ([Ouyang et al., 2023](https://arxiv.org/abs/2311.09656))
  - [Chain-of-thoughts](CoT) ([Wei et al., 202](https://arxiv.org/abs/2201.11903))
  - [Least-to-most prompting](Least-to-most) ([Zhou et al., 2022](https://arxiv.org/abs/2205.10625))
  - [Tree-of-Thoughts, BFS](ToT) ([Yao et al., 2023](https://arxiv.org/abs/2305.10601))
  - [Tree-of-Thoughts, DFS](ToT) ([Yao et al., 2023](https://arxiv.org/abs/2305.10601))
  - [Self-Eval Guided Decoding, Beam Search](Self-Eval) ([Xie et al., 2023](https://arxiv.org/abs/2305.00633))
  - [Grace Decoding](Grace) ([Khalifa et al., 2023](https://arxiv.org/abs/2305.14934))
  - [Eurus](Eurus) ([Yuan et al., 2024](https://arxiv.org/abs/2404.02078))
  - [PromptAgent](PromptAgent) ([Wang et al., 2023](https://arxiv.org/abs/2310.16427))
  - [DRPO](DRPO) ([Singla et al., 2024](https://aclanthology.org/2024.emnlp-main.1220/))

For each reasoning method, we provide scripts for different datasets, including GSM8K, AQuA, blocksworld, Prontoqa, StrategyQA, Game24, Crosswords, etc.

## How to run the examples
You can find the README about how to run the code under each directory with `cd examples/reasoning_method/dataset`.

## How to switch the base LLMs

Generally, all examples should be runnable with any choices of the base models (in `reasoners/lm`), e.g., `hugginface`, `llama3`, `exllama`, `openai`, `claude`, etc. Simply change the code of model loading `base_model = ...` and it should work.

Note that, when you switch to a new LLM, you may need to manually set the `eos_token_id` for it, so that the generation would stop when the LLM finishes answering the question. Failing to set the `eos_token_id` properly may result in slow generation and wrong answer parsing.

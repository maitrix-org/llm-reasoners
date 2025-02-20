![logo](assets/reasoners_icon.png#pic_center)


<p align="center">
  <a href="https://www.llm-reasoners.net/">Home</a>
  |
  <a href="https://arxiv.org/abs/2404.05221">Paper (COLM2024)</a>
  |
  <a href="https://www.llm-reasoners.net/blog">Blog</a>
  |
  <a href="https://discord.gg/PxDJby9W">Discord</a>
  |
  <a href="https://maitrix.org/">@Maitrix.org</a>
</p>

---

**LLM Reasoners** is a library designed to enhance LLMs' ability to perform complex reasoning using advanced algorithms. It provides:


- **Cutting-Edge Reasoning Algorithms**
  
  The library offer the most up-to-date search algorithms for reasoning with LLMs, such as:
  
  - [Reasoner Agent](example/ReasonerAgent-Web) ([Deng et al., 2025](https://reasoner-agent.maitrix.org/))
  - [Inference-time Scaling with PRM](examples/Inference-Scaling-SGL/math500) ([Snell et al., 2024](https://arxiv.org/abs/2408.03314))
  - [Reasoning-via-Planning, MCTS](examples/RAP) ([Hao et al., 2023](https://arxiv.org/abs/2305.14992))
  - [Tree-of-Thoughts, BFS](examples/ToT) ([Yao et al., 2023](https://arxiv.org/abs/2305.10601))
  
  <details>
    <summary>(Show more supported algorithms)</summary>
  
  - [StructChem](examples/StructChem) ([Ouyang et al., 2023](https://arxiv.org/abs/2311.09656))
  - [Chain-of-thoughts](examples/CoT) ([Wei et al., 2022](https://arxiv.org/abs/2201.11903))
  - [Least-to-most prompting](examples/Least-to-most) ([Zhou et al., 2022](https://arxiv.org/abs/2205.10625))
  - [Tree-of-Thoughts, DFS](examples/ToT) ([Yao et al., 2023](https://arxiv.org/abs/2305.10601))
  - [Self-Eval Guided Decoding, Beam Search](examples/Self-Eval) ([Xie et al., 2023](https://arxiv.org/abs/2305.00633))
  - [Grace Decoding](examples/Grace) ([Khalifa et al., 2023](https://arxiv.org/abs/2305.14934))
  - [Eurus](examples/Eurus) ([Yuan et al., 2024](https://arxiv.org/abs/2404.02078))
  - [PromptAgent](examples/PromptAgent) ([Wang et al., 2023](https://arxiv.org/abs/2310.16427))
  - [DRPO](examples/DRPO) ([Singla et al., 2024](https://aclanthology.org/2024.emnlp-main.1220/))
  
  </details>

- **Intuitive Visualization and Interpretation**: Our library provides a [visualization tool](https://www.llm-reasoners.net/) to aid users in comprehending the reasoning process. Even for complex reasoning algorithms like Monte-Carlo Tree Search, users can easily diagnose and understand the process with **one line of python code**. See an exmaple in the tutorial [notebook](demo.ipynb).
 
- **Efficient Reasoning with LLM**: Our library optimizes the performance of advanced reasoning techniques by integrating [SGLang](https://github.com/sgl-project/sglang), a high-performance LLM inference framework, featuring structured generation (Check out this [thread](https://x.com/MaitrixOrg/status/1885387184557199857) and [example](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/Inference-Scaling-SGL/math500)). We also support other LLM backends like `huggingface transformers`, `OpenAI API`, `Exllama`, `fairscale`, `llama.cpp`, etc.

- **Rigorous Implementation and Reproducibility**: We prioritize precision and reliability in our implementations, ensuring that our algorithms are not just theoretical concepts but practically usable tools. All methods implemented in LLM Reasoners are carefully engineered to be faithful to their original formulations and performance. It powers our [analysis](https://arxiv.org/abs/2404.05221) of reasoning algorithms published in COLM2024.

    <details>
    
    <summary> (Examples of Reproducibility) </summary>
    
    - LLM Reasoners has been tested to successfully reproduce the performance of [Tree-of-Thoughts](https://arxiv.org/abs/2305.10601), [Guided Decoding](https://arxiv.org/abs/2305.00633) and [GRACE Decoding](https://arxiv.org/abs/2305.14934) with their official implementation. We list the results reported in their paper / reproduced from their official repositories for reference (†). Some results are on the subsets of the first 100 examples (*).
    
    <div align="center">
        
    |Method|Base LLM|GSM8k|
    |--|--|--|
    |[Guided Decoding](https://arxiv.org/abs/2305.00633)<sup>†</sup>|CodeX (PAL)|0.80|-|-|-|-|-|
    |Guided Decoding|CodeX (PAL)|[0.83\*](examples/guided_gsm8k)|-|-|-|-|-|
    
    |Method|Base LLM|Game of 24|
    |--|--|--|
    |[Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)<sup>†</sup>|GPT-3.5-turbo|0.22|
    |Tree-of-Thoughts|GPT-3.5-turbo|[0.22](examples/tot_game24)|
    
    |Method|Base LLM|GSM8k|
    |--|--|--|
    |[GRACE Decoding](https://arxiv.org/abs/2305.14934)<sup>†</sup>|Flan-T5-Large (Fine-tuned)|0.34|-|-|-|-|-|
    |GRACE Decoding| Flan-T5-Large (Fine-tuned)|[0.33\*](examples/grace_gsm8k)|-|-|-|-|-|
    </div>
    
    </details>

## News
- Feb. 6, 2025: Thrilled to introduce **ReasonerAgent** - A fully open source, ready-to-run agent that does research 🧐 in a web browser and answers your queries. Check out this [thread](https://x.com/MaitrixOrg/status/1887584291087098063), and explore the [code](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/ReasonerAgent-Web) here! 
- Jan. 31, 2025: LLM Reasoners has integrated [SGLang](https://github.com/sgl-project/sglang). Enjoy 100x speed-up with a one-line change! New applications like PRM-guided search for inference-time scaling are also available. See more details in this [post](https://x.com/MaitrixOrg/status/1885387184557199857).
- Dec. 20, 2024: We now supported planning algorithms (MCTS, DFS/BFS, Beam Search) in web environments with [BrowserGym](https://github.com/ServiceNow/BrowserGym), check the [README](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/browsergym) to try out!

<details>

<summary>(Show more news)</summary>

- Nov. 13, 2024: We integrated [DRPO](https://aclanthology.org/2024.emnlp-main.1220/), a tuning-free alignment method published at EMNLP 2024 ([link](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/DRPO)).
  
- Jul. 10, 2024: Our paper on [LLM Reasoners](https://arxiv.org/abs/2404.05221) is accepted to [COLM 2024](https://colmweb.org/index.html)!
- Jun. 24, 2024: [PromptAgent](https://arxiv.org/abs/2310.16427) is in LLM Reasoners! Let it help you write down a super detailed prompt for your task ([here](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/PromptAgent)).
- May. 14, 2024: Check out [Eurus](https://arxiv.org/abs/2404.02078), a suit of LLMs optimized for reasoning. With LLM Reasoners, Eurus-RM can easily boost Llama-8B from 0.49 to 0.73 📈 on GSM8k ([code](examples/Eurus)).
- May. 2, 2024: We have integrated our first reasoning method for scientific reasoning, [StructChem](https://arxiv.org/abs/2311.09656)! Check it out [here](https://github.com/maitrix-org/llm-reasoners/tree/main/examples/StructChem).
- Apr. 22, 2024: We integrated [Llama-3](https://github.com/meta-llama/llama3), with additional useful APIs (e.g., customizing EOS tokens, calculating likelihood)
- **Apr. 8, 2024: Our new [paper](assets/Reasoners.pdf) introducing LLM Reasoners is available!**
- Mar. 29, 2024: [Grace Decoding](https://arxiv.org/abs/2305.14934) has been incoporated!
- Oct. 25, 2023: A [video tutorial](https://www.youtube.com/watch?v=5QfOxtiw_ZU) on the visualizer of LLM Reasoners are available.

- Oct. 23, 2023: Reasoning-via-Planning is accepted to EMNLP 2023! Check our [paper](https://arxiv.org/abs/2305.14992) with updated results and discussion!
</details>


## Introduction of the library

![Library Structure](assets/figure2_reasoners_v5.png)

We abstract an LLM reasoning algorithm into three key components, *reward function*, *world model*, and *search algorithm* (see the formulation in our [paper](https://arxiv.org/abs/2404.05221)), corresponding to three classes in the library, <tt>SearchConfig</tt>, <tt>WorldModel</tt> and <tt>SearchAlgorithm</tt> respectively. Besides, there are <tt>LLM APIs</tt> to power other modules, <tt>Benchmark</tt>, and <tt>Visualization</tt> to evaluate or debug the reasoning algorithm (middle). To implement a reasoning algorithm for a certain domain (a <tt>Reasoner</tt> object), a user may inherit the <tt>SearchConfig</tt> and <tt>WorldModel</tt> class, and import a pre-implemented <tt>SearchAlgorithm</tt>. We also show a concrete example of solving Blocksworld with RAP using LLM Reasoners (bottom).


## Quick Tour
Let's go through the code of reasoning over Blocksworld problems. Note that the code is simplified for demonstration (check [here](demo.ipynb) for a runnable notebook).

The first step is to define the world model: you will set up an initial state given a question in `init_state`, judge whether a state is terminal in `is_terminal`, and most importantly, define the world dynamics with `step`:
```python
from typing import NamedTuple
import utils
from reasoners import WorldModel, LanguageModel
import copy

BWState = str
BWAction = str

class BlocksWorldModel(WorldModel[BWState, BWAction]):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt

    def init_state(self) -> BWState:
        # extract the statement from a given problem
        # e.g., "the red block is clear, the blue block is clear..."
        return BWState(utils.extract_init_state(self.example)) 

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        # call the LLM to predict the state transition
        state = copy.deepcopy(state)
        # load the prompt for the LLM to predict the next state
        # e.g. "... I have that <state>, if I <action>, then ..."
        world_update_prompt = self.prompt["update"].replace("<state>", state).replace("<action>", action)
        world_output = self.base_model.generate([world_update_prompt],
                                    eos_token_id="\n", hide_input=True, temperature=0).text[0].strip()
        new_state = utils.process_new_state(world_output)
        # till now, we have the new state after the action
        # the following part is to speed up the reward calculation

        # we want to check the portion of the satisfied subgoals, and use it as a part of the reward
        # since we have predicted the new state already, we can just check it here at convenience
        goal_reached = utils.goal_check(utils.extract_goals(self.example, new_state))
        # return the new state and the additional dictionary (to be passed to the reward function)
        return new_state, {"goal_reached": goal_reached}

    def is_terminal(self, state: BWState) -> bool:
        # define the condition the terminal state to stop the search
        # e.g., all the subgoals are met
        if utils.goal_check(utils.extract_goals(self.example), state.blocks_state) == 1:
            return True
        return False
```
Then, it's time to consider how to search for the optimal reasoning chain. It involves `get_actions` to get the action space given a state, and the most important `reward` as the guidance for reasoning. For Monte-Carlo Tree Search, we can additionally define a `fast_reward` to speed up the roll-out stage.
```python
import utils
from world_model import BWState, BWAction
from reasoners import SearchConfig, LanguageModel
class BWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 reward_alpha=0.5,
                 goal_reward_default=0.,
                 goal_reached_reward=100) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        # some parameters to calculate the fast reward or reward (explained below)
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward

    def get_actions(self, state: BWState) -> list[BWAction]:
        # use a rule-based function to extract all legal actions
        return utils.generate_all_actions(state)

    def fast_reward(self, state: BWState, action: BWAction) -> tuple[float, dict]:
        # build an in-context learning prompt (similar to the one used in Chain-of-thoughts reasoning)
        inputs = self.prompt["icl"].replace("<init_state>", state)\
            .replace("<goals>", utils.extract_goals(self.example))
        # concatenate a candidate action after the prompt, and test its loglikelihood
        intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]
        # the reward is a combination of intuition and goal satisfaction
        # in fast_reward, we skip the calculation of goal satisfaction and use a default value
        fast_reward = intuition * self.reward_alpha + self.goal_reward_default * (1 - self.reward_alpha)
        # cache some information for the reward calculation later (will be passed to `reward` function)
        details = {'intuition': intuition}
        return fast_reward, details

    def reward(self, state: BWState, action: BWAction,
               intuition: float = None,
               goal_reached: tuple[bool, float] = None) -> float:
        # note that `intuition` (cached in `fast_reward`) and `goal_reached` (cached in `step`) are automatically passed as parameters to this reward function
        if goal_reached == 1:
            # if the goal state is reached, we will assign a large reward
            goal_reward = self.goal_reached_reward
        else:
            # otherwise assign the reward based on the portion of satisfied subgoals
            goal_reward = goal_reached
        # the reward is a combination of intuition and goal satisfaction
        reward = intuition * self.reward_alpha + goal_reward * (1 - self.reward_alpha)
        # return the reward and an additional dictionary (to be saved in the log for visualization later)
        return reward, {'intuition': intuition, 'goal_reached': goal_reached}
```
Now, we are ready to apply a reasoning algorithm to solve the problem:
```python
from reasoners.algorithm import MCTS
from reasoners.lm import LLaMAModel
from world_model import BlocksWorldModel
from search_config import BWConfig

llama_model = LLaMAModel(llama_ckpts, llama_size, max_batch_size=1)
with open(prompt_path) as f:
    prompt = json.load(f)
world_model = BlocksWorldModel(base_model=base_model, prompt=prompt)
config = BWConfig(base_model=llama_model, prompt=prompt)
# save the history of every iteration for visualization
search_algo = MCTS(output_trace_in_each_iter=True)
reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
for i, example in enumerate(dataset):
    algo_output = reasoner(example)
    # save the MCTS results as pickle files
    with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
        pickle.dump(algo_output, f)
```
Finally, we can easily visualize the reasoning process:
```python
import pickle
from reasoners.visualization import visualize
with open("logs/bw_MCTS/xxx/algo_output/1.pkl", 'rb') as f:
    mcts_result = pickle.load(f)

from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode

# by default, a state will be presented along with the node, and the reward with saved dictionary in `SearchConfig.reward` will be presented along with the edge. 
# we can also define a helper function to customize what we want to see in the visualizer.
def blocksworld_node_data_factory(n: MCTSNode) -> NodeData:
    return NodeData({"block state": n.state.blocks_state if n.state else None,
                     "satisfied": n.fast_reward_details if n.fast_reward_details else "Not expanded"})
def blocksworld_edge_data_factory(n: MCTSNode) -> EdgeData:
    return EdgeData({"reward": n.reward, "intuition": n.fast_reward_details["intuition"]})
visualize(mcts_result, node_data_factory=blocksworld_node_data_factory,
                       edge_data_factory=blocksworld_edge_data_factory)
```
Then a URL of the visualized results will pop up. The figure will be interactive and look like the examples shown on our [demo website](https://llm-reasoners.net/).
## Installation

Make sure to use Python 3.10 or later.

```bash
conda create -n reasoners python=3.10
conda activate reasoners
```

### Install from `pip`

```bash
pip install llm-reasoners
```

### Install from github
(Recommended if you want to run the examples in the github repo)

```bash
git clone https://github.com/Ber666/llm-reasoners --recursive
cd llm-reasoners
pip install -e .
```
Adding `--recursive` will help you clone exllama and LLM-Planning automatically. Note that some other optional modules may require other dependencies. Please refer to the error message for details.

## Citation
This project is an extension of the following paper:
```bibtex
@inproceedings{hao2023reasoning,
  title={Reasoning with Language Model is Planning with World Model},
  author={Hao, Shibo and Gu, Yi and Ma, Haodi and Hong, Joshua and Wang, Zhen and Wang, Daisy and Hu, Zhiting},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={8154--8173},
  year={2023}
}
@article{hao2024llm,
  title={LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models},
  author={Hao, Shibo and Gu, Yi and Luo, Haotian and Liu, Tianyang and Shao, Xiyan and Wang, Xinyuan and Xie, Shuhua and Ma, Haodi and Samavedhi, Adithya and Gao, Qiyue and others},
  journal={arXiv preprint arXiv:2404.05221},
  year={2024}
}
```

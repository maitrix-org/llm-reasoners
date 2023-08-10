![logo](images/image.png#pic_center)

---


**LLM Reasoners** is a library to enable LLMs to conduct complex reasoning, with advanced reasoning algorithms. It approaches multi-step reasoning as planning and searches for the optimal reasoning chain, which achieves the best balance of exploration vs exploitation with the idea of "World Model" and "Reward".

Given any reasoning problem, simply define the reward function and an optional world model (explained below), and let LLM reasoners take care of the rest, including Reasoning Algorithms, Visualization, LLM calling, and more!


## Why Choose LLM Reasoners?

- **Cutting-Edge Reasoning Algorithms**: We offer the most up-to-date search algorithms for reasoning with LLMs, such as [RAP-MCTS](https://arxiv.org/abs/2305.14992), [Tree-of-Thoughts](https://arxiv.org/abs/2305.10601), [Guided Decoding](https://arxiv.org/abs/2305.00633), and more. These advanced algorithms enable tree-structure reasoning and outperform traditional chain-of-thoughts approaches.

- **Intuitive Visualization and Interpretation**: Our library provides visualization tools to aid users in comprehending the reasoning process. Even for the most complex reasoning algorithms like Monte-Carlo Tree Search, users can easily diagnose and understand what occurred with one line of python code.

- **Compatibility with any LLM libraries**: Our framework is compatible with any LLM frameworks, e.g. Huggingface transformers, OpenAI API, etc. Specifically, we integrated LLaMA with the option of using [fairscale](https://github.com/facebookresearch/llama) backend for improved multi-GPU performance or [LLaMA.cpp](https://github.com/ggerganov/llama.cpp) backend with lower hardware requirements.


## Benchmarks
We tested different reasoning algorithms on the following benchmarks (to be updated). For Tree-of-thoughts and Guided Decoding, we also list the results reported in their paper /  reproduced from their official repositories for reference (†). Some results are on the subsets of the first 100 examples (*).

|Method|Base LLM|[GSM8K](https://arxiv.org/abs/2110.14168)|[AQuA](https://arxiv.org/abs/2008.12520)|[SVAMP](https://arxiv.org/abs/2103.07191)|[ASDiv](https://arxiv.org/abs/2106.15772)|[CommonsenseQA](https://arxiv.org/abs/1811.00937)|[StrategyQA](https://arxiv.org/abs/2101.02235)|
|-|-|-|-|-|-|-|-|
|[CoT](https://arxiv.org/abs/2201.11903)|LLaMA-33B|0.29|-|-|-|-|-|
|CoT+[SC](https://arxiv.org/abs/2203.11171)|LLaMA-33B|0.47|-|-|-|-|-|
|[Least-to-Most](https://arxiv.org/abs/2205.10625)+SC|LLaMA-33B|0.43|-|-|-|-|-|
|[Guided Decoding](https://arxiv.org/abs/2305.00633)<sup>†</sup>|CodeX (PAL)|0.80|-|-|-|-|-|
|Guided Decoding|CodeX (PAL)|[0.83\*](examples/guided_gsm8k)|-|-|-|-|-|
|[RAP](https://arxiv.org/abs/2305.14992)|LLaMA-33B|0.49|-|-|-|-|-|
|[RAP (aggr)](https://arxiv.org/abs/2305.14992)|LLaMA-33B|0.52|-|-|-|-|-|


|Method|Base LLM|[Blocksworld](https://arxiv.org/abs/2305.15771)|[Game of 24](https://arxiv.org/abs/2305.10601)|[Mini Crosswords](https://arxiv.org/abs/2305.10601)|[ProntoQA](https://arxiv.org/abs/2210.01240)|
|-|-|-|-|-|-|
|CoT|LLaMA-33B|0.03|-|-|0.65|
|[Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)<sup>†</sup>|GPT-3.5-turbo|-|0.22|-|-|
|Tree-of-Thoughts|GPT-3.5-turbo|-|[0.22](examples/tot_game24)|-|-|
|RAP|LLaMA-33B|0.64|-|-|0.79|


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

LLM Reasoners formulate **reasoning as planning** ([RAP](https://arxiv.org/abs/2305.14992)). Different from Chain-of-thoughts reasoning which autoregressively samples the next action, our goal is to **efficiently search in the reasoning space for the optimal reasoning chain**. To achieve this, two components need to be defined: a world model and a reward function.

- **World model** defines the state transition, formally $P(s_{i+1} | s_i, a_i)$. A default world model regards the partial solution as the state and simply appends a new action/thought to the state as the state transition (the same formulation of [Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)). However, you’ll have the option to design a better world model which predicts and keeps track of a more meaningful state (e.g., environment status, intermediate variable values, etc. Check [RAP](https://arxiv.org/abs/2305.14992) for more examples), thus enhancing the reasoning. For the example shown above, we can naturally define the state as the condition of blocks (e.g., the red block is on the yellow block...), and a world model is to predict the condition of blocks after every potential action.

- **Reward function** provides a criterion to evaluate a reasoning step. Ideally, a reasoning chain with a higher accumulated reward should be more likely to be correct. For the example shown above, we can reward actions based on the increased number of accomplished subgoals they lead to. Besides, the likelihood of LLMs generating the action can also be used as a reward, to give the search a good prior.


After we have the world model and reward function, it's time to apply an algorithm to search for the optimal reasoning trace. Here, we show the process of Monte-Carlo Tree Search:

![Alt text](images/mcts_animation.gif)

## Quick Tour
Let's go through the code of reasoning over Blocksworld problems. Note that the code is simplified for demonstration (check [here](https://github.com/Ber666/llm-reasoners/tree/main/examples/rap_blocksworld) for full experiment code).

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
Then an URL of the visualized results will pop up. The figure will be interactive and look like the examples shown in our [demo website](https://llm-reasoners.net/).
## Installation
Make sure to use Python 3.10 or later.

```bash
conda create -n reasoners python=3.10
conda activate reasoners
```

Clone the repository and install the package:
```bash
git clone https://github.com/Ber666/llm-reasoners
cd llm-reasoners
pip install -e .
```
Note that some optional modules may need other dependencies. Please refer to the error message for details.

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

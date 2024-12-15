# Web Agent Planning

This is an example of using LLM-Reasoners to perform Monte Carlo Tree Search (MCTS) and other planning methods (e.g. Greedy/Beam Search/DFS, etc.) on the [BrowserGym](https://github.com/ServiceNow/BrowserGym) environment. LLMs as agent policies are strong baselines, while the absolute performance on agent tasks are still far from human level. The inference-time planning (1) improves the policy's accuracy and (2) can be scaled up smoothly.

## Code Overview

BrowserGym offers an OpenAI gym-like interface for web environments, supporting benchmarks like Miniwob++, Webarena, and more. It simplifies web agent creation/testing by providing preloaded gym environments with task information and reward systems.

LLM-Reasoners enhance this setup with tree search algorithms, using LLMs to generate and evaluate actions. Besides LLM evaluations, the environment provides reward signals for node expansion.

- `inference_mcts.py`: Performs tree search on the environment w. MCTS. Currently it is a _open-loop_ planner, meaning it doesn't use a world model to predict the next state; instead, it directly interacts with the environment.
- `inference_dfs.py`: Performs tree search on the environment w. DFS.
- `inference_beam.py`: Performs tree search on the environment w. Beam Search.
- `gym_env.py`: Implements `EnvironmentGym`, wrapping the BrowserGym environment. `EnvironmentGym` functions like a `WorldModel`, using the environment for state transitions. Tree search requires careful backtracking, achieved by storing and replaying action histories, though this method is generic and applicable to any OpenAI gym-like environment.
- `search_config.py`: Defines `SearchConfigBrowsergym` for node generation/evaluation and reward calculation. This is the core of the tree search.
- `visualize.py`: Visualizes the search tree with saved search results in `.pickle` files.

## Setup

### Setup BrowserGym

```bash
git clone https://github.com/ServiceNow/BrowserGym.git
cd BrowserGym
make install
```

### Installing datasets (just install the ones you want to use)

**1. [Webarena](https://webarena.dev/) - Locally Hosted**

First set up the environment variables:

```bash
export BASE_URL="http://localhost"

# webarena environment variables (change ports as needed)
export WA_SHOPPING="$BASE_URL:8082/"
export WA_SHOPPING_ADMIN="$BASE_URL:8083/admin"
export WA_REDDIT="$BASE_URL:8080"
export WA_GITLAB="$BASE_URL:9001"
export WA_WIKIPEDIA="$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:443"
export WA_HOMEPAGE="$BASE_URL:80"

# if your webarena instance offers the FULL_RESET feature (optional)
export WA_FULL_RESET="$BASE_URL:7565"

# otherwise, be sure to NOT set WA_FULL_RESET, or set it to an empty string
export WA_FULL_RESET=""
```

Then host the webarena server following the [webarena docker instructions](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md).

When you finish, you can test the shopping instance by running

```bash
curl $WA_SHOPPING
```

**2. [VisualWebArena](https://jykoh.com/vwa) - Locally Hosted**

First, set up the environment variables:

```bash
export BASE_URL="http://localhost"

# visualwebarena environment variables (change ports as needed)
export VWA_CLASSIFIEDS="$BASE_URL:8083"
export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export VWA_SHOPPING="$BASE_URL:8082"
export VWA_REDDIT="$BASE_URL:8080"
export VWA_WIKIPEDIA="$BASE_URL:8081"
export VWA_HOMEPAGE="$BASE_URL:80"

# if your webarena instance offers the FULL_RESET feature (optional)
export VWA_FULL_RESET="$BASE_URL:7565"

# otherwise, be sure to NOT set VWA_FULL_RESET, or set it to an empty string
export VWA_FULL_RESET=""
```

Then, host the VisualWebArena server following the [visualwebarena docker instructions](https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md).

When you finish, you can test the shopping instance by running

```bash
curl $VWA_SHOPPING
```

### Misc Setup

Set up your LLM API keys and setup nltk

```bash
# Take OpenAI as an example. Feel free to use other LLMs.
export OPENAI_API_KEY="your_openai_api_key"
```

Download nltk punkt_tab if you haven't done so.

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## Run Tree Search

Run tree search (MCTS) on a web agent task.

```bash

python inference_mcts.py \
    --task_name <task_name, e.g. webarena.599> \
    --action_set <action_set, the action set to use, e.g. webarena> \
    --exp_dir <exp_dir, default results/tree-search> \
    --model <model, default gpt-4o-mini> \
    --n_iters <n_iters, default 10> \
    --depth_limit <depth_limit, default 10> \
```

Check more options in `inference_mcts.py`.

The search takes a few minutes to complete mainly consisting of the three following time consumptions:

1. LLM call as the policy and reward function
2. Environment interaction
3. Tree search expansion

For example, the default params above will take about 10 minutes to complete.

For DFS and Beam Search, you can run the respective files:

```bash
# DFS
python inference_dfs.py \
    --task_name <task_name> \
    --action_set <action_set> \
    --exp_dir <exp_dir> \
    --model <model> \
    --total_states <total_states> \
    --max_per_state <max_per_state> \
    --depth <depth> \

# Beam Search
python inference_beam.py \
    --task_name <task_name> \
    --action_set <action_set> \
    --exp_dir <exp_dir> \
    --model <model> \
    --beam_size <beam_size> \
    --max_depth <max_depth> \

```

## Visualize Search Tree

One key feature of LLM-Reasoners planner is we provide an online visualizer to smoothly visualize and debug the search tree.

```bash
python visualize.py \
    --task_name <task_name> \
    --exp_dir <exp_dir> \
```

If running successfully, you should see a visualizer link like this (hosted by LLM-Reasoners), e.g.,
https://main.d1puk3wdon4rk8.amplifyapp.com/visualizer/266f7660-0b9c-4cb8-96f3-1cd4aa719afa?accessKey=75503b6e

## Acknowledgements

Huge thanks to [WebArena](https://webarena.dev/) and [VisualWebArena](https://jykoh.com/vwa) for the amazing testbeds for web agent research.

Huge thanks to ServiceNow for providing the [BrowserGym](https://github.com/ServiceNow/BrowserGym) and the following [AgentLab](https://github.com/ServiceNow/AgentLab) (we're working in progress to integrate) infras for streamlining web agent task experiments.

# browsergym

https://github.com/ServiceNow/BrowserGym

## Overview

BrowserGym provides an OpenAI gym-like interface for interacting with web environments. It also comes with built in support for various web benchmarks such as Miniwob++, Webarena, VisualWebArena, WorkArena, etc. With browsergym, you can essentially provide a task name string (i.e. "miniwob.login-user", "webarena.599", etc.), and get a gym environment object preloaded with the task information along with systems to automatically analyze traces and provide rewards. This provides a great deal of convenience for creating/testing web agents.

For browsergym's step function, you pass in an action string (i.e. "click(element_id)"), and it will directly execute that code on the environment. It should be noted that the action code string provided is appended to another code string that contains all the function definitions (click, hover, fill, etc.) and that combined code is what is actually executed. These function definitions are generated from BrowserGym's HighLevelActionSet class. Currently, the action space is the minimal set needed for WebArena, so if more actions are needed, specify them during the instantiation of HighLevelActionSet.

LLM-Reasoners is here to provide tree search algorithms for exploring this environment. At every step, SearchConfigBrowsergym, uses a LLM to generate a list of possible actions, and then also uses a LLM to evaluate the quality of those actions. This is identical to most examples in LLM-Reasoners. The only main difference is that on top of the LLM's evaluation (fast reward), the environment also provides a reward signal, which needs to be considered when calculating the reward after expanding a node.

EnvironmentGym essentially takes on the same functionality as a WorldModel, but instead of relying on an LLM for the state transitions, you can just use the environment directly.

Since tree search is being directly performed on the environment, backtracking becomes more complicated. Before taking an action on a state, you have to make sure that the environment is aligned. This is currently done by additionally storing the action history in the state tuples, and then resetting + replaying the actions. This is inefficient, but it is generic and should work for any openai gym-like environment. Depending on the task, there may be more efficient ways to handle this.

gym_env.py - Contains EnvironmentGym, an implementation of the Environment class. Mainly a wrapper class for the BrowserGym environment.
searchconfig.py - Contains SearchConfigBrowsergym, which defines how to generate/evaluate nodes in tree search. Also defines how to calculate fast_reward (sa pair hasn't been expanded), and reward (sa pair has been expanded).
visualize.py - Script to visualize the search tree. For convenience, some MCTSResult pickle objects have been provided in the results/tree-search directory.

## Setting up browsergym

```bash
git clone https://github.com/ServiceNow/BrowserGym.git
cd BrowserGym
make install
```

## Installing datasets

### 1. Miniwob++

https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md

```bash
cd ./examples/browsergym
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git

cd miniwob-plusplus/miniwob/html/miniwob
export MINIWOB_URL="file://$(pwd)/"
```

### 2. Webarena - Locally Hosted

https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/webarena/README.md

```bash
# download nltk punkt_tab
python -c "import nltk; nltk.download('punkt_tab')"

# set up webarena environment variables
export BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770"
export WA_SHOPPING_ADMIN="$BASE_URL:7780"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
export WA_WIKIPEDIA="$BASE_URL:8898/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:3000"
export WA_HOMEPAGE="$BASE_URL:4399"

```

Then serve the webarena server following the [webarena instructions](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md). e.g., for the shopping instance (note the .tar file needs downloading), run

```bash
docker load --input shopping_final_0712.tar
docker run --name shopping -p 7770:80 -d shopping_final_0712
# wait ~1 min to wait all services to start

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="${WA_SHOPPING}" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="${WA_SHOPPING}" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush
```

Test the shopping instance by running

```bash
curl $WA_SHOPPING
```

After that, you should be able to run

```bash
python wa_test.py
```

to run a webarena task.

## API Keys

```bash
# i.e. openai
export OPENAI_API_KEY="your_openai_api_key"
```

## Running Reasoning-as-Planning (RAP) on a Task

```bash

python inference.py <task_name>

```

## Visualize Search Tree

Some MCTSResult pickle objects have been provided in the results/tree-search directory.

```bash
# python visualize.py <task_name>
python visualize.py webarena.599
```

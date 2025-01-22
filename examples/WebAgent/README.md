# Web Agent Example:

This is an example code for running the web agent implemented by LLM Reasoners. We also offer a baseline agent based on `BrowsingAgent` from [OpenHands](https://github.com/All-Hands-AI/OpenHands).

## Setup

Aside from installing `reasoners`, please also install the dependencies specific to this example using the command below:

```
pip install -r requirements.txt
```

## Datasets

We provide three datasets for evaluating web agents as informational assistants: 
1. [FanOutQA](https://fanoutqa.com/index.html), which requires the agent to answer questions that require searching for and compiling information from multiple websites. We include their development set of 310 examples. 
2. FlightQA, a dataset prepared by us to evaluate the ability of LLM agents in answering queries with varying number of constraints, specifically while searching for live flight tickets using the internet. To control for confounding variables like specific query content, we iteratively add to lists of constraints to form new questions. In total we have 120 examples consisted of 20 groups of questions ranging from 3 to 8 constraints.
3. [WebArena](https://webarena.dev), which comprises benchmarking tasks on a few self-hosted websites, including information seeking, site navigation and content management. 

## Commands

To run evaluation using one of our datasets, use the following command as an example:

```
python main.py \
    [job_name] \
    --dataset [fanout, flightqa, webarena] \
    --agent [reasoner, openhands] \
    --config_name [optional, only applies to reasoner agent; 
                   options: browsergym, browsergym_world_model, opendevin, opendevin_llama, opendevin_world_model, opendevin_webarena, opendevin_webarena_world_model, browsergym_webarena, browsergym_webarena_world_model
                   default: browsergym] \
    --model [optional, any model accessible via litellm; default: gpt-4o] \
    --api_key [optional, by default set to content in file 'llm-reasoners/examples/WebAgent/default_api_key.txt' if exists, otherwise None] \
    --start_idx [optional, index of the first example; default: 0] \
    --end_idx [optional, index of the last example; default: 9999999] \
    --shuffle [store_true, whether to shuffle the dataset before slicing with start_idx and end_idx] \
    --seed [optional, used as seed when --shuffle is set; default: 42] \
    --max_steps [optional, maximum steps the agent can take on a single task; default: 30] \
    --output_dir [optional, location to store browsing interaction data; default: ./browsing_data] 
```

One way to speed up the inference is to open several terminals and run inference on separate slices of the data.

Before that, you'll need to enter your default API key at `default_api_key.txt`.

The agent outputs will be stored under `browsing_data` and will be automatically loaded for evaluation.

## Evaluation

### FanOutQA and FlightQA
```
python evaluation/fanout/run.py [job_name]

python evaluation/flight/run.py [job_name]
```

Note: Running evaluation for FlightQA involves calling `gpt-4o`, which may incur costs to you. So please run judiciously.

### WebArena
Running WebArena requires setup following [this guide](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#pre-installed-amazon-machine-image-recommended). Then the evaluation can be run with the [script](evaluation/webarena/run_inference.sh) provided.

## Installation

If any issue arises while trying to install BrowserGym, please refer to the [official repo](https://github.com/ServiceNow/BrowserGym).

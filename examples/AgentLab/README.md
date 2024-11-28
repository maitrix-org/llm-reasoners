
<div align="center">
    


[![pypi](https://badge.fury.io/py/agentlab.svg)](https://pypi.org/project/agentlab/)
[![PyPI - License](https://img.shields.io/pypi/l/agentlab?style=flat-square)]([https://opensource.org/licenses/MIT](http://www.apache.org/licenses/LICENSE-2.0))
[![PyPI - Downloads](https://img.shields.io/pypi/dm/agentlab?style=flat-square)](https://pypistats.org/packages/agentlab)
[![GitHub star chart](https://img.shields.io/github/stars/ServiceNow/AgentLab?style=flat-square)](https://star-history.com/#ServiceNow/AgentLab)
[![Code Format](https://github.com/ServiceNow/AgentLab/actions/workflows/code_format.yml/badge.svg)](https://github.com/ServiceNow/AgentLab/actions/workflows/code_format.yml)
[![Tests](https://github.com/ServiceNow/AgentLab/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/ServiceNow/AgentLab/actions/workflows/unit_tests.yml)



[🛠️ Setup](#%EF%B8%8F-setup-agentlab) &nbsp;|&nbsp; 
[🤖 Assistant](#-ui-assistant) &nbsp;|&nbsp; 
[🚀 Launch Experiments](#-launch-experiments) &nbsp;|&nbsp;
[🔍 Analyse Results](#-analyse-results) &nbsp;|&nbsp;
<br>
[🏆 Leaderboard](#-leaderboard) &nbsp;|&nbsp; 
[🤖 Build Your Agent](#-implement-a-new-agent) &nbsp;|&nbsp;
[↻ Reproducibility](#-reproducibility) 


<img src="https://github.com/user-attachments/assets/47a7c425-9763-46e5-be54-adac363be850" alt="agentlab-diagram" width="700"/>


[Demo solving tasks:](https://github.com/ServiceNow/BrowserGym/assets/26232819/e0bfc788-cc8e-44f1-b8c3-0d1114108b85)


</div>

AgentLab is a framework for developing and evaluating agents on a variety of
[benchmarks](#-supported-benchmarks) supported by
[BrowserGym](https://github.com/ServiceNow/BrowserGym).

AgentLab Features:
* Easy large scale parallel [agent experiments](#-launch-experiments) using [ray](https://www.ray.io/)
* Building blocks for making agents over BrowserGym
* Unified LLM API for OpenRouter, OpenAI, Azure, or self-hosted using TGI.
* Preferred way for running benchmarks like WebArena
* Various [reproducibility features](#reproducibility-features)
* Unified [LeaderBoard](https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard)

## 🎯 Supported Benchmarks

| Benchmark | Setup  <br> Link | # Task <br> Template| Seed  <br> Diversity | Max  <br> Step | Multi-tab | Hosted Method | BrowserGym <br> Leaderboard |
|-----------|------------|---------|----------------|-----------|-----------|---------------|----------------------|
| [WebArena](https://webarena.dev/) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/webarena/README.md) | 812 | None | 30 | yes | self hosted (docker) | soon |
| [WorkArena](https://github.com/ServiceNow/WorkArena) L1 | [setup](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started) | 33 | High | 30 | no | demo instance | soon |
| [WorkArena](https://github.com/ServiceNow/WorkArena) L2 | [setup](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started) | 341 | High | 50 | no | demo instance | soon |
| [WorkArena](https://github.com/ServiceNow/WorkArena) L3 | [setup](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started) | 341 | High | 50 | no | demo instance | soon |
| [WebLinx](https://mcgill-nlp.github.io/weblinx/) | - | 31586 | None | 1 | no | self hosted (dataset) | soon |
| [VisualWebArena](https://github.com/web-arena-x/visualwebarena) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/visualwebarena/README.md) | 910 | None | 30 | yes | self hosted (docker) | soon |
| [AssistantBench](https://assistantbench.github.io/) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/assistantbench/README.md) | 214 | None | 30 | yes | live web | soon |
| [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) (soon) | - | - | None | - | - | live web | soon |
| [Mind2Web-live](https://huggingface.co/datasets/iMeanAI/Mind2Web-Live) (soon) | - | - | None | - | - | live web | soon |
| [MiniWoB](https://miniwob.farama.org/index.html) | [setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md) | 125 | Medium | 10 | no | self hosted (static files) | soon |


## 🛠️ Setup AgentLab

```bash
pip install agentlab
```

If not done already, install Playwright:
```bash
playwright install
```

Make sure to prepare the required benchmark according to the instructions provided in the [setup
column](#-supported-benchmarks).

```bash
export AGENTLAB_EXP_ROOT=<root directory of experiment results>  # defaults to $HOME/agentlab_results
export OPENAI_API_KEY=<your openai api key> # if openai models are used
```

<details>
<summary>Setup OpenRouter API</summary>

```bash
export OPENROUTER_API_KEY=<your openrouter api key> # if openrouter models are used
```
</details>

<details>
<summary>Setup Azure API</summary>

```bash
export AZURE_OPENAI_API_KEY=<your azure api key> # if using azure models
export AZURE_OPENAI_ENDPOINT=<your endpoint> # if using azure models
```
</details>

## 🤖 UI-Assistant 

Use an assistant to work for you (at your own cost and risk).

```bash
agentlab-assistant --start_url https://www.google.com
```

Try your own agent: 

```bash
agentlab-assistant --agent_config="module.path.to.your.AgentArgs"
```

## 🚀 Launch experiments

```python
# Import your agent configuration extending bgym.AgentArgs class
# Make sure this object is imported from a module accessible in PYTHONPATH to properly unpickle
from agentlab.agents.generic_agent import AGENT_4o_MINI 

from agentlab.experiments.study import make_study

study = make_study(
    benchmark="miniwob",  # or "webarena", "workarnea_l1" ...
    agent_args=[AGENT_4o_MINI],
    comment="My first study",
)

study.run(n_jobs=5)
```

Relaunching incomplete or errored tasks

```python
from agentlab.experiments.study import Study
study = Study.load("/path/to/your/study/dir")
study.find_incomplete(include_errors=True)
study.run()
```

See [main.py](main.py) to launch experiments with a variety of options. This is like a lazy CLI that
is actually more convenient. Just comment and uncomment the lines you need or modify at will (but
don't push to the repo).


### Job Timeouts

The complexity of the wild web, Playwright, and asyncio can sometimes cause jobs to hang. This
disables workers until the study is terminated and relaunched. If you are running jobs sequentially
or with a small number of workers, this could halt your entire study until you manually kill and
relaunch it. In the Ray parallel backend, we've implemented a system to automatically terminate jobs
exceeding a specified timeout. This feature is particularly useful when task hanging limits your
experiments. 

### Debugging

For debugging, run experiments with `n_jobs=1` and use VSCode's debug mode. This allows you to pause
execution at breakpoints.

### About Parallel Jobs

Running one agent on one task corresponds to a single job. Conducting ablation studies or random
searches across hundreds of tasks with multiple seeds can generate more than 10,000 jobs. Efficient
parallel execution is therefore critical. Agents typically wait for responses from the LLM server or
updates from the web server. As a result, you can run 10–50 jobs in parallel on a single computer,
depending on available RAM.

⚠️ **Note for (Visual)WebArena**: These benchmarks have task dependencies designed to minimize
"corrupting" the instance between tasks. For example, an agent on task 323 could alter the instance
state, making task 201 impossible. To address this, the Ray backend accounts for task dependencies,
enabling some degree of parallelism. On WebArena, you can disable dependencies to increase
parallelism, but this might reduce performance by 1–2%.

⚠️ **Instance Reset for (Visual)WebArena**: Before evaluating an agent, the instance is
automatically reset, a process that takes about 5 minutes. When evaluating multiple agents, the
`make_study` function returns a `SequentialStudies` object to ensure proper sequential evaluation of
each agent. AgentLab currently does not support evaluations across multiple instances, but you could
either create a quick script to handle this or submit a PR to AgentLab. For a smoother parallel
experience, consider using benchmarks like WorkArena instead.

## 🔍 Analyse Results

### Loading Results

The class [`ExpResult`](https://github.com/ServiceNow/BrowserGym/blob/da26a5849d99d9a3169d7b1fde79f909c55c9ba7/browsergym/experiments/src/browsergym/experiments/loop.py#L595) provides a lazy loader for all the information of a specific experiment. You can use [`yield_all_exp_results`](https://github.com/ServiceNow/BrowserGym/blob/da26a5849d99d9a3169d7b1fde79f909c55c9ba7/browsergym/experiments/src/browsergym/experiments/loop.py#L872) to recursively find all results in a directory. Finally [`load_result_df`](https://github.com/ServiceNow/AgentLab/blob/be1998c5fad5bda47ba50497ec3899aae03e85ec/src/agentlab/analyze/inspect_results.py#L119C5-L119C19) gathers all the summary information in a single dataframe. See [`inspect_results.ipynb`](src/agentlab/analyze/inspect_results.ipynb) for example usage.

```python
from agentlab.analyze import inspect_results

# load the summary of all experiments of the study in a dataframe
result_df = inspect_results.load_result_df("path/to/your/study")

# load the detailed results of the 1st experiment
exp_result = bgym.ExpResult(result_df["exp_dir"][0])
step_0_screenshot = exp_result.screenshots[0]
step_0_action = exp_result.steps_info[0].action
```


### AgentXray

https://github.com/user-attachments/assets/06c4dac0-b78f-45b7-9405-003da4af6b37

In a terminal, execute:
```bash
agentlab-xray
```

You can load previous or ongoing experiments in the directory `AGENTLAB_EXP_ROOT` and visualize
the results in a gradio interface.

In the following order, select:
* The experiment you want to visualize
* The agent if there is more than one
* The task
* And the seed

Once this is selected, you can see the trace of your agent on the given task. Click on the profiling
image to select a step and observe the action taken by the agent.


**⚠️ Note**: Gradio is still developing, and unexpected behavior has been frequently noticed. Version 5.5 seems to work properly so far. If you're not sure that the proper information is displaying, refresh the page and select your experiment again.


## 🏆 Leaderboard

Official unified [leaderboard](https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard) across all benchmarks. 

Experiments are on their way for more reference points using GenericAgent. We are also working on code to automatically push a study to the leaderboard.

## 🤖 Implement a new Agent

Get inspiration from the `MostBasicAgent` in
[agentlab/agents/most_basic_agent/most_basic_agent.py](src/agentlab/agents/most_basic_agent/most_basic_agent.py).
For a better integration with the tools, make sure to implement most functions in the
[AgentArgs](src/agentlab/agents/agent_args.py#L5) API and the extended `bgym.AbstractAgentArgs`.

If you think your agent should be included directly in AgenLab, let us know and it can be added in
agentlab/agents/ with the name of your agent.  

## ↻ Reproducibility
Several factors can influence reproducibility of results in the context of evaluating agents on
dynamic benchmarks.

### Factors affecting reproducibility
* **Software version**: Different versions of Playwright or any package in the software stack could
  influence the behavior of the benchmark or the agent.
* **API-based LLMs silently changing**: Even for a fixed version, an LLM may be updated e.g. to
  incorporate the latest web knowledge.
* **Live websites**:
  * WorkArena: The demo instance is mostly fixed in time to a specific version but ServiceNow
    sometimes pushes minor modifications.
  * AssistantBench and GAIA: These rely on the agent navigating the open web. The experience may
    change depending on which country or region, some websites might be in different languages by
    default.
* **Stochastic Agents**: Setting the temperature of the LLM to 0 can reduce most stochasticity.
* **Non-deterministic tasks**: For a fixed seed, the changes should be minimal

### Reproducibility Features
* `Study` contains a dict of information about reproducibility, including benchmark version, package
  version and commit hash
* The `Study` class allows automatic upload of your results to
  [`reproducibility_journal.csv`](reproducibility_journal.csv). This makes it easier to populate a
  large amount of reference points. For this feature, you need to `git clone` the repository and install via `pip install -e .`.
* **Reproduced results in the leaderboard**. For agents that are reprocudibile, we encourage users
  to try to reproduce the results and upload them to the leaderboard. There is a special column
  containing information about all reproduced results of an agent on a benchmark.
* **ReproducibilityAgent**: [You can run this agent](src/agentlab/agents/generic_agent/reproducibility_agent.py) on an existing study and it will try to re-run
  the same actions on the same task seeds. A visual diff of the two prompts will be displayed in the
  AgentInfo HTML tab of AgentXray. You will be able to inspect on some tasks what kind of changes
  between the two executions. **Note**: this is a beta feature and will need some adaptation for your
  own agent.


## Misc

if you want to download HF models more quickly
```
pip install hf-transfer
pip install torch
export HF_HUB_ENABLE_HF_TRANSFER=1
```

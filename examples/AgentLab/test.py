from src.agentlab.experiments.study import make_study
from src.agentlab.agents.reasoners_agent.agent_configs import AGENT_4o_MINI

# export AGENTLAB_EXP_ROOT=$HOME/Documents/agentlab/agentlab_results
# export MINIWOB_URL="file://$HOME/Documents/browsergym/miniwob-plusplus/miniwob/html/miniwob/"

study = make_study(
    benchmark="miniwob",  # or "webarena", "workarnea_l1" ...
    agent_args=[AGENT_4o_MINI],
    comment="My first study",
    tiny_test_task_names=["miniwob.login-user"]
)

study.run(n_jobs=1)

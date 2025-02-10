import argparse

from browsergym.experiments.loop import EnvArgs, ExpArgs

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.exp_utils import RESULTS_DIR
from agentlab.experiments.launch_exp import import_object


def make_exp_args(agent_args: AgentArgs, start_url="https://www.google.com"):

    try:
        agent_args.flags.action.demo_mode = "default"
    except AttributeError:
        pass

    if isinstance(agent_args, GenericAgentArgs):
        agent_args.flags.enable_chat = True

    exp_args = ExpArgs(
        agent_args=agent_args,
        env_args=EnvArgs(
            max_steps=1000,
            task_seed=None,
            task_name="openended",
            task_kwargs={
                "start_url": start_url,
            },
            headless=False,
            record_video=True,
            wait_for_user_message=True,
            viewport={"width": 1500, "height": 1280},
            slow_mo=1000,
        ),
    )

    return exp_args


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent_config",
        type=str,
        default="agentlab.agents.generic_agent.AGENT_4o_MINI",
        help="""Python path to the agent config. Defaults to : "agentlab.agents.generic_agent.AGENT_4o".""",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="The start page of the agent. Defaults to https://www.google.com",
    )

    args, unknown = parser.parse_known_args()
    agent_args = import_object(args.agent_config)
    exp_args = make_exp_args(agent_args, args.start_url)
    exp_args.prepare(RESULTS_DIR / "ui_assistant_logs")
    exp_args.run()


if __name__ == "__main__":
    main()

import argparse
import pickle

from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from reasoners.algorithm.mcts import MCTSNode
from browsergym.core.action.parsers import highlevel_action_parser

from examples.browsergym.gym_env import StateGym


def browsergym_node_data_factory(n: MCTSNode) -> NodeData:
    return NodeData(
        {"expanded": "expanded"}
        if n.state is not None
        else {"expanded": "not expanded"}
    )


def browsergym_edge_data_factory(n: MCTSNode) -> EdgeData:
    function_calls = highlevel_action_parser.search_string(n.action)
    function_calls = sum(function_calls.as_list(), [])

    python_code = ""
    for function_name, function_args in function_calls:
        python_code += (
            function_name
            + "("
            + ", ".join([repr(arg) for arg in function_args])
            + ")\n"
        )

    return EdgeData(
        {
            "Q": n.Q,
            "self_eval": n.fast_reward_details["self_eval"],
            "action": python_code,
        }
    )


def load_and_visualize(task_name: str):
    result = pickle.load(open(f"./results/tree-search/{task_name}/result.pkl", "rb"))

    visualize(
        result,
        node_data_factory=browsergym_node_data_factory,
        edge_data_factory=browsergym_edge_data_factory,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the result of a tree search task."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task to visualize.",
    )
    args = parser.parse_args()

    load_and_visualize(args.task_name)

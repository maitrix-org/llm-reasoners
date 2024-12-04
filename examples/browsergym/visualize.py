import pickle
import numpy as np
import argparse
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode


def process_obs_for_viz(obs: dict[str, any]):
    """Process the observation for visualization"""
    # Hard code special observation fields to be serialized
    # TODO: use a better expresssion, e.g., image
    obs.update(
        {
            k: str(obs[k])[:50]
            for k in ["screenshot", "axtree_txt", "pruned_html"]
            if k in obs
        }
    )

    # Convert tuple/array fields (keeping the iteration as needed)
    for k, v in obs.items():
        if isinstance(v, tuple) or isinstance(v, np.ndarray):
            obs[k] = list(v)

    # Convert int64 active_page_index to int to be serialized
    if "active_page_index" in obs:
        obs["active_page_index"] = [int(x) for x in obs["active_page_index"]]

    # Extract clean action history from the whole action history string
    if "action_history" in obs:
        obs["clean_action_history"] = list(
            map(parse_action_from_proposal_string, obs["action_history"])
        )

    # Extract clean action from the last action string
    if "last_action" in obs:
        obs["clean_last_action"] = parse_action_from_proposal_string(obs["last_action"])

    return obs


def parse_action_from_proposal_string(proposal: str):
    """Extract the action from the proposal string wrapped in triple backticks"""
    import re

    match = re.search(r"```(.+?)```", proposal)
    return match.group(1) if match else proposal


def browsergym_node_data_factory(x: MCTSNode):
    """Generate the node data for the tree visualization"""
    if not x.state:
        return {}
    # last_obs = process_obs_for_viz(x.state.last_obs)
    current_obs = process_obs_for_viz(x.state.current_obs)
    return {
        "step_idx": x.state.step_idx,
        "action_history": x.state.action_history,
        "reward": x.state.reward,
        "terminated": x.state.terminated,
        "truncated": x.state.truncated,
        **current_obs,
    }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Visualize search tree from pickle file"
    )
    parser.add_argument(
        "--tree_log_file",
        type=str,
        required=True,
        help="Path to the tree log pickle file",
    )

    # Parse arguments
    args = parser.parse_args()

    # Load the pickle file
    mcts_result = pickle.load(open(args.tree_log_file, "rb"))

    visualize(mcts_result, node_data_factory=browsergym_node_data_factory)


if __name__ == "__main__":
    main()

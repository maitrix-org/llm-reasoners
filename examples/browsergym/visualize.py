import argparse
import pickle
import numpy as np
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from typing import Union
from reasoners.algorithm.mcts import MCTSNode
from reasoners.algorithm.beam_search import BeamSearchNode
from reasoners.algorithm.dfs import DFSNode
from browsergym.core.action.parsers import highlevel_action_parser
from gym_env import StateGym


from PIL import Image
import io
import base64

from PIL import Image, UnidentifiedImageError
import io
import base64


def compress_base64_image(base64_str, output_format="JPEG", quality=50):
    try:
        # Determine the prefix
        prefix = ""
        if base64_str.startswith("data:image"):
            prefix, base64_str = base64_str.split(",", 1)

        # Decode the base64 string to bytes
        image_data = base64.b64decode(base64_str)

        # Open the image using PIL
        image = Image.open(io.BytesIO(image_data))

        # Create a BytesIO object to hold the compressed image
        compressed_image_io = io.BytesIO()

        # Save the image to the BytesIO object with the desired compression
        image.save(compressed_image_io, format=output_format, quality=quality)

        # Get the compressed image bytes
        compressed_image_bytes = compressed_image_io.getvalue()

        # Encode the compressed image bytes back to a base64 string
        compressed_base64_str = base64.b64encode(compressed_image_bytes).decode("utf-8")

        # Add the prefix back if it was present
        if prefix:
            compressed_base64_str = f"{prefix},{compressed_base64_str}"

        return compressed_base64_str

    except (base64.binascii.Error, UnidentifiedImageError) as e:
        print(f"Error processing image: {e}")
        return None


def process_obs_for_viz(obs: dict[str, any], verbose: bool = False):
    """Process the observation for visualization"""
    processed_obs = {}

    # Convert tuple/array fields (keeping the iteration as needed)
    for k, v in obs.items():
        if isinstance(v, tuple) or isinstance(v, np.ndarray):
            obs[k] = list(v)
        processed_obs[k] = v

    # Truncate the long text fields to 50 characters
    processed_obs.update(
        {k: str(obs[k])[:50] for k in ["axtree_txt", "pruned_html"] if k in obs}
    )
    # Convert int64 active_page_index to int to be serialized
    if "active_page_index" in obs:
        processed_obs["active_page_index"] = [int(x) for x in obs["active_page_index"]]
    # Extract clean action history from the whole action history string
    if "action_history" in obs:
        processed_obs["clean_action_history"] = list(
            map(simple_parse_action_from_string, obs["action_history"])
        )
    # Extract clean action from the last action string
    if "last_action" in obs:
        processed_obs["clean_last_action"] = simple_parse_action_from_string(
            obs["last_action"]
        )

    # FIXME: the screenshot is too large to be uploaded to the visualizer server; uncomment this when the issue is fixed
    # processed_obs["screenshot"] = compress_base64_image(processed_obs["screenshot"])
    processed_obs["screenshot"] = str(processed_obs["screenshot"])[:50]

    if not verbose:
        return {
            "screenshot": processed_obs["screenshot"],
            "last_action": processed_obs["clean_last_action"],
        }

    return processed_obs


def simple_parse_action_from_string(proposal: str):
    """Extract the action from the proposal string wrapped in triple backticks"""
    import re

    match = re.search(r"```(.+?)```", proposal)
    return match.group(1).strip() if match else proposal


def browsergym_node_data_factory(x: MCTSNode, verbose: bool = False):
    """Generate the node data for the tree visualization"""
    if not x.state:
        return {}
    current_obs = process_obs_for_viz(x.state.current_obs, verbose)

    if not verbose:
        return {
            "step_idx": int(x.state.step_idx),
            "reward": x.state.reward,
            **current_obs,
        }
    else:
        return {
            "step_idx": int(x.state.step_idx),
            "action_history": x.state.action_history,
            "reward": x.state.reward,
            "terminated": x.state.terminated,
            "truncated": x.state.truncated,
            **current_obs,
        }


def browsergym_edge_data_factory(
    n: Union[MCTSNode, BeamSearchNode, DFSNode], verbose: bool = False
) -> EdgeData:
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

    if isinstance(n, MCTSNode):
        return EdgeData(
            {
                "Q": n.Q,
                "self_eval": n.fast_reward_details["self_eval"],
                "action": python_code,
            }
        )
    else:
        return EdgeData(
            {
                "reward": n.reward,
                "action": python_code,
            }
        )


def load_and_visualize(args):
    result = pickle.load(open(f"{args.exp_dir}/{args.task_name}/result.pkl", "rb"))

    visualize(
        result,
        node_data_factory=lambda x: browsergym_node_data_factory(x, args.verbose),
        edge_data_factory=lambda x: browsergym_edge_data_factory(x, args.verbose),
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
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results/tree-search",
        help="The directory to save the visualization results.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Whether to print the visualization results.",
    )
    args = parser.parse_args()

    load_and_visualize(args)

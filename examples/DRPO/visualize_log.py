import pickle
from reasoners.visualization import visualize
from search_algo.beam_search import BeamSearchNode
from reasoners.algorithm.beam_search import BeamSearchResult as reasoner_bs
with open("logs/mistral-7b-chain/algo_output/output.pkl", 'rb') as f:
    beam_result = pickle.load(f)
    
beam_result = reasoner_bs(
    terminal_node=beam_result.terminal_node,
    cum_reward=beam_result.cum_reward,
    terminal_state=beam_result.terminal_state,
    tree=beam_result.tree,
    trace=beam_result.trace
)

from reasoners.visualization.tree_snapshot import NodeData,EdgeData
from search_algo.beam_search import BeamSearchNode
# by default, a state will be presented along with the node, and the reward with saved dictionary in `SearchConfig.reward` will be presented along with the edge. 
# we can also define a helper function to customize what we want to see in the visualizer.
def blocksworld_node_data_factory(n: BeamSearchNode) -> NodeData:
    return NodeData({"System Prompt": n.state[-1].system_prompt if n.state else None})
def blocksworld_edge_data_factory(n: BeamSearchNode) -> EdgeData:
    return EdgeData({"reward": n.reward if n.reward is not None else None})
visualize(beam_result, node_data_factory=blocksworld_node_data_factory,
                       edge_data_factory=blocksworld_edge_data_factory)
import pickle
import sys
sys.path.append('..')
import os
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
mcts_result = pickle.load(open('/data/haotian/RAP_tune/llm-reasoners/logs/AQuA_clean_MCTS/10202023-063340/algo_output/5.pkl', 'rb'))
# for child in mcts_result.tree_state.children:
#     print(child.state[-1].answer_list)
print(mcts_result.terminal_state[-2].answer_list)
def gsm_node_data_factory(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
visualize(mcts_result, node_data_factory=gsm_node_data_factory)
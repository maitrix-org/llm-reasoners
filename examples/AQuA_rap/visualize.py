import pickle
import sys
sys.path.append('..')
import os
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
mcts_result = pickle.load(open('/data/haotian/RAP_tune/llm-reasoners/logs/AQuA_clean_MCTS/12012023-193559/algo_output/7.pkl', 'rb'))
print(mcts_result.terminal_state[-1].answer_list)
print(mcts_result.terminal_state[-1].answer_values)
print(len(mcts_result.terminal_state[-1].answer_list))
dict_values = mcts_result.terminal_state[-1].answer_values
length = 0
for i in dict_values:
    length += len(i)
print(length)
# print(mcts_result.aggregated_result)
def gsm_node_data_factory(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
visualize(mcts_result, node_data_factory=gsm_node_data_factory)
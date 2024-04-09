import pickle
import sys
sys.path.append('..')
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize")
    parser.add_argument("--pickle_dir", type=str)
    args = parser.parse_args()
    mcts_result = pickle.load(open(args.pickle_dir, 'rb'))
    print(f"answer_list: {mcts_result.terminal_state[-1].answer_list}")
    print(f"answer_values: {mcts_result.terminal_state[-1].answer_values}")
    print(f"answer_list length: {len(mcts_result.terminal_state[-1].answer_list)}")
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
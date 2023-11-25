import itertools
from typing import Sequence, Any

from tqdm import tqdm
import pickle

from dataset import ProntoQADataset, ProntoQAProblem, ProntoQAExample
from reasoners import LanguageModel, SearchAlgorithm, Reasoner

from reasoners.lm import llama_cpp_model
from reasoners.visualization import visualize
from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAState, ProntoQAAction
from reasoners.algorithm import MCTS


class ProntoQAEvaluator:
    def __init__(self, dataset: ProntoQADataset, search_algo: SearchAlgorithm) -> None:
        dataset_list = list(itertools.islice(dataset, 1))
        self.queries = [obj.test_example.query.split(':', 1)[1].strip() for obj in dataset_list]
        self.dataset = iter(dataset_list)
        self.answers = [obj.test_example.answer for obj in dataset_list]
        # self.dataset = itertools.islice(dataset,1) # TODO: remove slicing
        self.language_model = language_model
        self.world_model = ProntoQAWorldModel(base_model=language_model)
        self.search_config = ProntoQAConfig(base_model=language_model)
        self.search_algo = search_algo
        self.reasoner: Reasoner[ProntoQAState, ProntoQAAction, ProntoQAExample] = Reasoner(
            world_model=self.world_model,
            search_config=self.search_config,
            search_algo=self.search_algo
        )
        # self.queries = [obj.test_example.query.split(':',1)[1].strip() for obj in list(self.dataset)]

    def evaluate(self) -> Sequence[Any]:
        return list(tqdm(map(
            self.reasoner,
            self.dataset
        )))


if __name__ == '__main__':
    """
    For testing purposes only. Will be removed.
    """

    # language_model = llama_cpp_model.LlamaCppModel(
    #     path='/data/adithya/llama/llama-2-13b-chat/llama-2-13b-chat.Q4_0.gguf'
    # )
    import torch, os
    import numpy as np
    # device = torch.device("cuda:6")
    from reasoners.lm import ExLlamaModel 
    language_model = ExLlamaModel(os.environ['LLAMA2_CKPTS'],
                                None, 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=None,
                                log_output=True)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

    dataset = ProntoQADataset.from_file(
        'examples/rap_prontoqa/data/345hop_random_true.json'
    )
    
    evaluator = ProntoQAEvaluator(dataset=dataset, search_algo=MCTS(w_exp=2.5,n_iters=15,output_trace_in_each_iter=True, cum_reward=np.mean))
    # print(list(dataset))
    # search_algorithm =MCTS()

    print(evaluator.queries)
    result = evaluator.evaluate()
    print(len(result))
    correct=0
    n_test = 1
    for i in range(n_test):
        print(f"{i}: {result[i]}")
        with open(f"logs/pronto_debug/MCTS_{i}.pkl", 'wb') as file:
            # Use pickle to dump the object to the file
            pickle.dump(result[i], file)

        
        if(result[i].terminal_state is None):
            print(f"{i} th query: {evaluator.queries[i] == None}")
        else:
            print(f"{i} th query: {evaluator.queries[i] == result[i].terminal_state.last_state}")
            print(f"result[i].trace_of_nodes[-2].body : {result[i].trace_of_nodes[-2]}")
            if evaluator.answers[i] == result[i].terminal_state.body:
                correct+=1
        
    print(f"accuracy: {(correct*100)/n_test}") 

    from reasoners.visualization.tree_snapshot import NodeData, EdgeData
    from reasoners.algorithm.mcts import MCTSNode
    
    '''
    def prontoqa_node_data_factory(n: MCTSNode) -> NodeData:
        return NodeData({"state": n.state.body if n.state else None})


    def prontoqa_edge_data_factory(n: MCTSNode[ProntoQAState, ProntoQAAction,ProntoQAExample]) -> EdgeData:
        return EdgeData({"Q": n.Q,"reward": n.reward, "action": n.action})


    visualize(result[0], node_data_factory=prontoqa_node_data_factory,
              edge_data_factory=prontoqa_edge_data_factory)
    '''
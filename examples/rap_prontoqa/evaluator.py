import itertools
import os
from typing import Sequence, Any
import json
from tqdm import tqdm
import pickle

from dataset import ProntoQADataset, ProntoQAProblem, ProntoQAExample
from reasoners import LanguageModel, SearchAlgorithm, Reasoner

from reasoners.lm import llama_cpp_model
from reasoners.visualization import visualize
from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAState, ProntoQAAction
from reasoners.algorithm import MCTS
from reasoners.benchmark import ProntoQAEvaluatorFinal

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# class ProntoQAEvaluator:
#     def __init__(self, dataset: ProntoQADataset, search_algo: SearchAlgorithm) -> None:
#         dataset_list = list(dataset)
#         self.queries = [obj.test_example.query.split(':', 1)[1].strip() for obj in dataset_list]
#         self.dataset = iter(dataset_list)
#         self.answers = [obj.test_example.answer for obj in dataset_list]
#         self.language_model = language_model
#         self.world_model = ProntoQAWorldModel(base_model=language_model)
#         self.search_config = ProntoQAConfig(base_model=language_model)
#         self.search_algo = search_algo
#         self.reasoner: Reasoner[ProntoQAState, ProntoQAAction, ProntoQAExample] = Reasoner(
#             world_model=self.world_model,
#             search_config=self.search_config,
#             search_algo=self.search_algo
#         )
#         # self.queries = [obj.test_example.query.split(':',1)[1].strip() for obj in list(self.dataset)]

#     def evaluate(self) -> Sequence[Any]:
#         return list(tqdm(map(
#             self.reasoner,
#             self.dataset
#         )))


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
                                mem_map=[16,22],
                                log_output=True)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

    dataset = ProntoQADataset.from_file(
        'data/345hop_random_true.json'
    )

    with open('data/example_next_steps.json') as f:
            init_prompt = json.load(f)
    
    # evaluator = ProntoQAEvaluator(dataset=dataset, search_algo=MCTS(w_exp=2.5,n_iters=15,output_trace_in_each_iter=True, cum_reward=np.mean))



    world_model = ProntoQAWorldModel(base_model=language_model)
    search_config = ProntoQAConfig(base_model=language_model)
    search_algo = MCTS(w_exp=1.5,n_iters=15,output_trace_in_each_iter=True, cum_reward=np.mean)
    reasoner =  Reasoner(
            world_model=world_model,
            search_config=search_config,
            search_algo=search_algo
        )

    evaluator = ProntoQAEvaluatorFinal(init_prompt=init_prompt['next_steps'],
                               sample_prompt_type="rap",
                               disable_log=False,
                               disable_tqdm=False, dataset = ProntoQADataset.from_file(
        'data/345hop_random_true.json'
    )
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4 ,log_dir="results/")
    print(f"accuracy: {accuracy}")
 


    # result = evaluator.evaluate()
    # correct=0
    # n_test = 5

    # directory_path = "logs/pronto_debug"
    # create_directory_if_not_exists(directory_path)

   
    # for i in range(n_test):
    #     print(f"{i}: {result[i]}")
    #     file_path = os.path.join(directory_path, f"MCTS_{i}.pkl")
    #     with open(file_path, 'wb') as file:
    #         # Use pickle to dump the object to the file
    #         pickle.dump(result[i], file)

        
    #     if(result[i].terminal_state is None):
    #         print(f"{i} th query: {evaluator.queries[i] == None}")
    #     else:
    #         print(f"{i} th query: {evaluator.queries[i] == result[i].terminal_state.last_state}")
    #         print(f"result[i].trace_of_nodes[-2].body : {result[i].trace_of_nodes[-2]}")
    #         if evaluator.answers[i] == result[i].terminal_state.body:
    #             correct+=1
        
    # print(f"accuracy: {(correct*100)/n_test}") 

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
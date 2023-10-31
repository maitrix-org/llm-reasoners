import itertools
from typing import Sequence, Any

from tqdm import tqdm

from dataset import ProntoQADataset, ProntoQAProblem, ProntoQAExample
from reasoners import LanguageModel, SearchAlgorithm, Reasoner
from reasoners import algorithm
from reasoners.lm import llama_cpp_model
from reasoners.visualization import visualize
from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAState, ProntoQAAction


class ProntoQAEvaluator:
    def __init__(self, dataset: ProntoQADataset, search_algo: SearchAlgorithm) -> None:
        self.dataset = itertools.islice(dataset,1) # TODO: remove slicing
        self.language_model = language_model
        self.world_model = ProntoQAWorldModel(base_model=language_model)
        self.search_config = ProntoQAConfig(world_model=self.world_model)
        self.search_algo = search_algo
        self.reasoner: Reasoner[ProntoQAState, ProntoQAAction, ProntoQAExample] = Reasoner(
            world_model=self.world_model,
            search_config=self.search_config,
            search_algo=self.search_algo
        )

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
    import torch
    # device = torch.device("cuda:6")
    from reasoners.lm import ExLlamaModel 
    language_model = ExLlamaModel('/data/haotian/RAP_tune/Llama-2-70B-GPTQ', 
                                None, 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=[16, 21])#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

    dataset = ProntoQADataset.from_file(
        '/data/adithya/345hop_random_true.json'
    )

    evaluator = ProntoQAEvaluator(dataset=dataset, search_algo=algorithm.MCTS())

    search_algorithm = algorithm.MCTS()

    result = evaluator.evaluate()

    print(result)

    from reasoners.visualization.tree_snapshot import NodeData, EdgeData
    from reasoners.algorithm.mcts import MCTSNode


    def prontoqa_node_data_factory(n: MCTSNode) -> NodeData:
        return NodeData({"state": n.state.body if n.state else None})


    def blocksworld_edge_data_factory(n: MCTSNode[ProntoQAState, ProntoQAAction]) -> EdgeData:
        return EdgeData({"reward": n.reward, "action": n.action})


    visualize(result[0], node_data_factory=prontoqa_node_data_factory,
              edge_data_factory=blocksworld_edge_data_factory)

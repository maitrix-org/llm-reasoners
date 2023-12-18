from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Mapping

import json
from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Union


@dataclass
class ProntoQAProblem:
    question: str
    query: str
    chain_of_thought: List[str]
    answer: str


@dataclass
class ProntoQAExample:
    in_context_examples: Mapping[str, ProntoQAProblem]
    test_example: ProntoQAProblem = field(default=None)
    """
    ProntoQAProblem(question='Every cat is a feline. Mammals are vertebrates. Bilaterians are animals. Vertebrates are chordates. Carnivores are mammals. Mammals are not cold-blooded. Each chordate is a bilaterian. Every feline is a carnivore. Snakes are cold-blooded. Animals are not unicellular. Every carnivore is not herbivorous. Fae is a cat.', query='True or false: Fae is not cold-blooded.', chain_of_thought=['Fae is a cat.', 'Every cat is a feline.', 'Fae is a feline.', 'Every feline is a carnivore.', 'Fae is a carnivore.', 'Carnivores are mammals.', 'Fae is a mammal.', 'Mammals are not cold-blooded.', 'Fae is not cold-blooded.'], answer='True')"""



@dataclass
class ProntoQADataset:
    examples: Dict[str, ProntoQAExample] = field(default_factory=dict)
    # "example1": ..., "example2": ..., ...

    @classmethod
    def from_file(cls, file_path: str) -> 'ProntoQADataset':
        instance = cls()
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        for example_key, example_value in raw_data.items():

            all_examples = {
                k: ProntoQAProblem(
                    question=e["question"],
                    query=e["query"],
                    chain_of_thought=e["chain_of_thought"],
                    answer=e["answer"]
                ) for k, e in example_value.items()
            }

            test_example = all_examples.pop('test_example', None)
            in_context_examples = []

            instance.examples[example_key] = ProntoQAExample(in_context_examples, test_example)

        return instance

    def __iter__(self) -> Iterator[Union[ProntoQAProblem, ProntoQAExample]]:
        return iter(self.examples.values())


# Sample usage
if __name__ == "__main__":
    pronto_qa_dataset = ProntoQADataset.from_file('/Users/xiyan/Downloads/345hop_random_true.json')

    for example in pronto_qa_dataset:
        print(example)
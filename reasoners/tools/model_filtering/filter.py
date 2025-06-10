import transformers
import abc
from typing import Dict


class Filter(abc.ABC):
    """
    Filter class for filtering data.
    """
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def check(self, data_entry: Dict) -> bool:
        pass


class LengthFilter(Filter):
    """
    Filter class for filtering data by length.
    """
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer = None, min_length: int = 0, max_length: int = 2048, length_tolerance: int = 100):
        if tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        else:
            self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length
        self.length_tolerance = length_tolerance
    
    def check(self, data_entry: Dict) -> bool:
        if data_entry["prompt"]:
            
            prompt_tokens = self.tokenizer.tokenize(self.tokenizer.apply_chat_template(data_entry["prompt"], tokenize=False))
        elif data_entry["raw_prompt"]:
            prompt_tokens = self.tokenizer.tokenize(data_entry["raw_prompt"])
        else:
            raise ValueError("No prompt found in data")
        # print(f"Prompt length: {len(prompt_tokens)}")
        return self.min_length <= len(prompt_tokens) <= self.max_length - self.length_tolerance

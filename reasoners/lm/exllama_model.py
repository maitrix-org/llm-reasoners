from .. import LanguageModel,GenerateOutput
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.lora import ExLlamaLora
import torch
from typing import Tuple, Union, Optional
import warnings
import copy
import sys
import numpy as np
import optimum
import os
from optimum.bettertransformer import BetterTransformer
from tqdm import tqdm

class ExLlamaModel(LanguageModel):
    def __init__(self, model_dir, tokenizer_dir, device, max_batch_size, max_new_tokens):
        super().__init__()
        tokenizer_path = os.path.join(model_dir, "tokenizer.model")
        model_config_path = os.path.join(model_dir, "config.json")
        st_pattern = os.path.join(model_dir, "*.safetensors")
        
from typing import Union, Optional
import warnings
import copy

from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig, AutoConfig, AutoModel
import torch
from peft import PeftModel
import numpy as np
# from optimum.bettertransformer import BetterTransformer
from accelerate import infer_auto_device_map, dispatch_model



class HFModelReward():
    def __init__(self, model_pth, tokenizer_pth, device='cuda:0', max_batch_size=1, max_new_tokens=None, max_length=2048, quantized=None, peft_pth=None, load_awq_pth=None,device_map=None, **kwargs):
        super().__init__()
        """
        Initializes a new instance of the `HFModel` class.

        Args:
            model_pth (str): The path to the directory containing the pre-trained model.
            tokenizer_pth (str): The path to the directory containing the pre-trained tokenizer.
            device (str): The device to use for running the model (e.g. "cpu", "cuda").
            max_batch_size (int, optional): The maximum batch size to use for inference. Defaults to 1.
            max_new_tokens (int, optional): The maximum number of new tokens to generate during inference. Defaults to None.
            max_length (int, optional): The maximum length of the input sequence. Defaults to 2048.
            quantized (str, optional): The type of quantization to use for the model. Can be "8bit", "nf4", "fp4", or "awq". Defaults to None.
            peft_pth (str, optional): The path to the directory containing the pre-trained PEFT model. Defaults to None.
            load_awq_pth (str, optional): The path to the directory containing the pre-trained AWQ model. Defaults to None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, lagacy=False, trust_remote_code=True)

        if quantized == "int8":
            print("int8 quantizing.............................")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModel.from_pretrained(
                model_pth,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto" if device_map is None else device_map,                                    
            )
        elif quantized == "nf4" or quantized  == "fp4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=quantized,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            print("quantizing.............................")

            self.model = AutoModel.from_pretrained(
                model_pth,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto" if device_map is None else device_map,
                
            )
        
        elif quantized == "awq":
            try:
                from awq.quantize.pre_quant import apply_awq
                from awq.quantize.quantizer import real_quantize_model_weight
            except ImportError as e:
                print(f'\033[31mError\033[0m: You need to install package awq to use {quantized=}. '
                      'It can be installed with \033[1mpip install -e .[awq]\033[0m under cloned reaonsers repo. '
                      'Refer to https://github.com/mit-han-lab/llm-awq for more details.')
                raise e
            config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True, lagacy=False)
            kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
            self.model = AutoModel.from_pretrained(model_pth, config=config, trust_remote_code=True,**kwargs)
            self.model.eval()
            awq_results = torch.load(load_awq_pth, map_location="cpu")
            apply_awq(self.model, awq_results)
            q_config = {
                "zero_point": True,  # by default True
                "q_group_size": 128,  # whether to use group quantization
            }
            real_quantize_model_weight(self.model, w_bit=4, q_config=q_config)
            kwargs = {"max_memory": None}
            device_map = infer_auto_device_map(self.model,no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"], **kwargs)
            self.model = dispatch_model(self.model, device_map=device_map)

        else:
            self.model = AutoModel.from_pretrained(
                model_pth,
                device_map="auto",
                trust_remote_code=True
            )
        if peft_pth is not None:
            self.model = PeftModel.from_pretrained(
                self.model, 
                peft_pth,
                torch_dtype=torch.float16
            )
        
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.device = device
        # self.model = BetterTransformer.transform(self.model) #not updated yet
        self.model.eval()
        # for old llama tokenizer's config, below is necessary
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        # if torch.__version__ >= "2" and sys.platform != "win32":#need to figure out this line
        #     self.model = torch.compile(self.model) ###make the faketensor bug, an on-going issue in pytorch
    
    
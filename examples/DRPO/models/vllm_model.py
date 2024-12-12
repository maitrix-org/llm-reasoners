from reasoners import LanguageModel as Model
from typing import Union, Optional, List
from vllm import LLM, SamplingParams
import numpy as np
from huggingface_hub import login
class VLLMModel(Model):
    def __init__(self, 
                 model_name,
                 num_gpus: int = 1,
                 dtype: str = 'bfloat16',
                 gpu_memory_utilization: float = 0.98,
                 max_model_len: Union[int, None] = None,
                 **kwargs
    ):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **kwargs
        )
        self.tokenizer = self.model.get_tokenizer()
    
    def generate(self, 
                 prompts, 
                 temperature: float = 0.7, 
                 top_p: float = 0.95,
                 max_new_tokens: int = 256,
                 stop: Optional[Union[str, List[str]]] = None,
                 **kwargs
    ) -> Union[str, list]:
        if isinstance(prompts, str):
            prompts = [prompts]
            
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop,
            **kwargs
        )
        
        outputs = self.model.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        if len(outputs) == 1:
            return outputs[0].outputs[0].text
        else:
            return [output.outputs[0].text for output in outputs]

    def get_next_token_logits(self,
                            prompt: Union[str, list[str]],
                            candidates: Union[list[str], list[list[str]]],
                            **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                          prompt: Union[str, list[str]],
                          **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")
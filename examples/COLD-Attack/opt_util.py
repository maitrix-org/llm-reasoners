import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
        use_cache=True,
    )


    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
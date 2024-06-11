import torch
import torch
import numpy as np
from util import *
# nltk.download('punkt')
import os
import openai

def compute_ppl_line(model, tokenizer, line):
    line = line.strip()
    line_ = tokenizer.encode(line)
    line_t = torch.tensor(line_, dtype=torch.long).cuda()
    loss = model(input_ids=line_t, labels=line_t).loss
    loss = loss.detach().clone().data.cpu().numpy()
    ppl = np.exp(loss)
    return ppl

def calculate_coverage(output_ln, key_words):
    x_words = word_tokenize(x)
    x_words = set(x_words)
    count = len(x_words.intersection(key_words))
    ratio = count / len(key_words)
    return ratio

def lora_to_base(model):
    try:
        model.base_model.disable_adapter_layers()
    except:
        print("No adapter layers to disable")
    model.eval()
    
def base_to_lora(model):
    try:
        model.base_model.enable_adapter_layers()
    except:
        print("No adapter layers to enable")
    model.train()

def tb_loss(log_pf, log_r, logz):
    loss = (log_pf + logz - log_r) ** 2
    return loss.mean()



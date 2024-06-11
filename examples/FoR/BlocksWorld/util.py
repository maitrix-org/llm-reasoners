import torch
import torch
import numpy as np
from util import *
# nltk.download('punkt')
import os

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

def tb_loss(log_pf, log_r, logz, log_bf=None, logpartition=False):
    print("log_pf: ", log_pf)
    print("log_r: ", log_r)
    print("logz: ", logz)
    if logpartition:
        if log_bf != None:
            scores = log_pf - log_r - log_bf
            loss = (scores - scores.mean()) ** 2 
        else:
            scores = log_pf - log_r
            loss = (scores - scores.mean()) ** 2 
    else:
        if log_bf != None:
            loss = (log_pf + logz - log_r - log_bf) ** 2
        else:
            loss = (log_pf + logz - log_r) ** 2
    return loss.mean()


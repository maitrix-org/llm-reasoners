import os
import numpy as np
import time
import wandb
import argparse
import random
import sys

from opt_util import *
from util import *

def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--world-model-base", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--if-zx", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--step", type=int, default=2)
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")

    parser.add_argument("--load-checkpoint-path", type=str, default=None, help="The trained gflownet")
    parser.add_argument("--train-data", type=str, default=None, help="The trained gflownet")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="The trained gflownet")
    parser.add_argument("--test-only", type=bool, default=False)
    parser.add_argument("--use-4bit", type=bool, default=True)
    parser.add_argument("--use-lora", type=bool, default=True)
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=25)
    parser.add_argument("--buffer_size", type=int, default=50)
    parser.add_argument("--epoch_nums", type=int, default=0)
    parser.add_argument("--test_sample_nums", type=int, default=10)
    parser.add_argument("--use-buffer-prob", type=float, default=0.5)
    parser.add_argument("--do_test", action="store_true", help="")
    parser.add_argument("--do_train", action="store_true", help="")
    parser.add_argument("--do_val", action="store_true", help="") 
    parser.add_argument("--do-sft", action="store_true", help="")
    
    parser.add_argument("--mode", type=str, default='game24',
                        choices=['blocksworld','game24'])
    parser.add_argument("--use-gpt-value", action='store_true') 
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15, help="maximum length of optimized logits.")
    parser.add_argument("--max-length", type=int, default=20, help="maximum length of complete sentence.")
    parser.add_argument("--logZ_init", type=int, default=0, help="initialization of logZ")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--no-loss-rerank", action="store_true", help="")
    parser.add_argument("--use-sysprompt", action="store_true", help="")
    # temperature
    parser.add_argument("--input-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    # lr
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate in the backward pass.")
    parser.add_argument("--logZ_lr", type=float, default=1e-5, help="learning rate in the backward pass.")
    # iterations
    parser.add_argument("--num-iters", type=int, default=10000)
    
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = options()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if args.seed != -1:
        seed_everything(args.seed)
    # Load pretrained model with lora
    model, tokenizer = load_model(args, device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    from game24 import game24_planning
    game24_planning(model,tokenizer,args)
        

if __name__ == "__main__":
    main()

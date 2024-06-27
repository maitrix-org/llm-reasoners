#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import argparse
import random
import sys
import logging
import torch
sys.path.insert(0, './GPT2ForwardBackward')

from nltk.corpus import stopwords
from opt_util import *
from util import *

stop_words = set(stopwords.words('english'))

def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--pretrained_model", type=str, default="/meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--world_model", type=str, default="/meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--step", type=int, default=2)
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--load-checkpoint-path", type=str, default=None, help="The trained gflownet")
    parser.add_argument("--test-only", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--use-4bit", type=bool, default=True)
    parser.add_argument("--use-lora", type=bool, default=True)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=50)
    parser.add_argument("--use-buffer-prob", type=float, default=0.5)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--ll-weight", type=float, default=1.0)
    parser.add_argument("--task", type=str, default='blocksworld',
                        choices=['blocksworld', 'alfworld'])
    parser.add_argument("--PG", type=bool, default=False,
                        )
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--logZ_init", type=float, default=5, help="initialization of logZ")
    # temperature
    parser.add_argument("--reward_temp_start", type=float, default=1.0,
                        help="temperature of reward")
    parser.add_argument("--reward_temp_end", type=float, default=2.0,
                        help="temperature of reward")
    parser.add_argument("--epsilon_start", type=float, default=0.50,
                        help="epsilon greedy")
    parser.add_argument("--epsilon_end", type=float, default=0.01,
                        help="epsilon greedy")
    parser.add_argument("--pf_temp_start", type=float, default=2.0,
                        help="temperature of reward")
    parser.add_argument("--pf_temp_end", type=float, default=1.0,
                        help="temperature of reward")
    parser.add_argument("--pf_temp_prob", type=float, default=0.5,
                        help="probability of using tempered policy")
    parser.add_argument("--p_buffer_start", type=float, default=0.45,
                        help="probability of using tempered policy")
    parser.add_argument("--p_buffer_end", type=float, default=0.5,
                        help="probability of using tempered policy")
    
    # lr
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate in the backward pass.")
    parser.add_argument("--logZ_lr", type=float, default=1e-5, help="learning rate in the backward pass.")
    # iterations
    parser.add_argument("--num-iters", type=int, default=10000)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
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

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def main():
    args = options()
    torch.set_float32_matmul_precision('medium')
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if args.seed != -1:
        seed_everything(args.seed)
    # Load pretrained model with lora
    logger = get_logger('logging.log')
    logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info(args)
    model, tokenizer = load_model(args, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "blocksworld" in args.task:
        from blocksworld import blocksworld_planning
        blocksworld_planning(model, tokenizer, device, args)
    else:
        pass

if __name__ == "__main__":
    main()

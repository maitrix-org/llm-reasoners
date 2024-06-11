import pandas as pd
import os

from nltk.corpus import stopwords

from util import *
from bw_utils import *
stop_words = set(stopwords.words('english'))
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from replay_buffer import ReplayBuffer
import os
import yaml
import sys
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
#from Executor import Executor
from util import *
from pathlib import Path
import torch
from typing import Tuple
import json
import time
import warnings

from lightning_module import *
from lightning_data import *


warnings.filterwarnings("ignore")

def blocksworld_planning(model, tokenizer, device, args, model_back=None):
    rbuffer = ReplayBuffer(buffer_size=args.buffer_size) # initialize a replay buffer
    logZ = torch.nn.Parameter(torch.tensor([0], dtype=torch.float))
    data = PromptDataModule(
        args=args,
        tokenizer=tokenizer,
        train_size=0.1,
        limit_prompts=None,
    )

    train_probes = data.train_data
    val_probes = data.val_data
    test_probes = data.test_data


    task = BlocksWorldGFNTask(
        args=args, 
        model=model,
        logZ=logZ, 
        tokenizer=tokenizer,
        replay_buffer=rbuffer,
        train_data=train_probes,
        val_data=val_probes,
        test_data=test_probes)

    checkpoint_callback = ModelCheckpoint(
            filename="ckpt_{epoch:02d}_",
            every_n_epochs=1,
            save_top_k=-1,  # <--- this is important!
        )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],     
        accelerator="gpu",
        devices=1,
        precision="16-true",
        #precision=16,
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_checkpointing=True,
        accumulate_grad_batches=10,
    )
    trainer.fit(model=task, datamodule=data) 
    model.save_pretrained("./weight")
    trainer.test(model=task, datamodule=data)
    trainer.test(ckpt_path="./last", dataloaders=data.test_dataloader())

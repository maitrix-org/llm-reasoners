from util import *
from bw_utils import *

import pytorch_lightning as pl
from replay_buffer import ReplayBuffer
import sys
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import torch
import warnings
import pickle
from lightning_module_selection import *
from lightning_data import *


warnings.filterwarnings("ignore")


def get_problem(instance, domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain)
    return reader.parse_instance(instance)

def blocksworld_planning(model, tokenizer, device, args, model_back=None):

    rbuffer = ReplayBuffer(buffer_size=args.buffer_size) # initialize a replay buffer

    logZ = torch.nn.Parameter(torch.tensor([args.logZ_init], dtype=torch.float))

    data = PromptDataModule(
        args=args,
        tokenizer=tokenizer,
        train_size=0.4,
        limit_prompts=None,
    )
    trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
        precision="16-true",
        max_epochs=args.epochs,
        check_val_every_n_epoch=10,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        profiler="simple",
        enable_checkpointing=False
        )
    train_probes = data.train_data
    val_probes = data.val_data
    with trainer.init_module():
        task = BlocksWorldGFNTask(
            args, 
            model,
            logZ, 
            tokenizer,
            rbuffer,
            train_data=train_probes,
            val_data=val_probes)

    trainer.fit(model=task, datamodule=data)
    transition_path = f"/transitions/{args.step}/transition.pkl"
    with open(transition_path, 'wb') as f:
        pickle.dump(task.transitions, f)

    model.save("./ckpt/6-step")
    print("PEFT saved...")
    trainer.test(ckpt_path="last", dataloaders=data.test_dataloader())

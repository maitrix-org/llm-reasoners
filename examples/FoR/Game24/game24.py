import pandas as pd
from replay_buffer import ReplayBuffer
from lightning_data import *
from lightning_module import *
import pytorch_lightning as pl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def game24_planning(model,tokenizer,args):
    rbuffer = ReplayBuffer(buffer_size=args.buffer_size) #
    data = Game24DataModule(
        args=args,
        tokenizer=tokenizer,
        train_size=0.8,
        limit_prompts=None
    )
    train_data = list(pd.read_csv('data/24.csv')['Puzzles'])
    logZ = torch.nn.Parameter(torch.tensor([0], dtype=torch.float))
    task = Game24GTNTask(
        args,
        model,
        logZ,
        tokenizer,
        replay_buffer=rbuffer,
        train_data=train_data,
        val_data=train_data
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision=16,
        max_epochs=args.epoch_nums,
        num_sanity_val_steps=0,
    )
    
    if args.do_train:
        trainer.fit(model=task, datamodule=data)
        if args.do_test:
            trainer.test(model=task, datamodule=data)
    elif args.do_test:
        print("Start Test")
        trainer.test(model=task, datamodule=data,ckpt_path=args.load_checkpoint_path)

from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
import sys

from util import *
from bw_utils import *

import yaml
import json



class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        tokenizer,
        train_size=0.8,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")

        self.base_prompt = json.load(open("data/prompt/next_step_examples.json", 'r'))["input"]

        
        self.tokenizer = tokenizer
        self.args = args
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def setup(self, stage):
        train_data = []
        all_data = []
        all_data = json.load(open("data/345hop_random_true.json", 'r'))
        ood = ["3", "31", "131071", "real", "number", "imaginary", "numbers"]
        indis = ["bony", "insect", "cold-blooded", "animal"]
        for key in all_data:
            if key.startswith("example"):
         
                ACTIONS = all_data[key]["test_example"]["question"]
                QUERY = all_data[key]["test_example"]["query"]
                PLAN = all_data[key]["test_example"]["chain_of_thought"]
                GT = all_data[key]["test_example"]["answer"]
            if any([o in ACTIONS for o in ood]):
                if len(self.test_data) < 120:
                    self.val_data.append([ACTIONS, QUERY, PLAN, GT])
                    
            if any([o in ACTIONS for o in indis]):
         
                train_data.append([ACTIONS, QUERY, PLAN, GT]) 
                
                if len(self.val_data) < 120:
                    self.val_data.append([ACTIONS, QUERY, PLAN, GT])

        cot_path = "sft/data.txt"

        train_answer = []
        valid_answer = []
        with open(cot_path, "r") as f:
            for line in f.read().splitlines():
                answer = line[line.index("answer='")+len("answer='"):line.index("';")]
               
                if "correct=True" in line:
                    train_answer.append(answer)
                else:
                    valid_answer.append(answer)

        for ans in train_answer:
            actions = ans.split("\\n")
            i = len(actions) - 1
            for train_ex in train_data:
                start_id = 0
                while i >= 0 and train_ex[2][-1-start_id*2] == actions[i]:
                    start_id += 1
                    i -= 1
                if i == -1 and len(self.train_data) < 20:
                    self.train_data.append(train_ex) 
                    break

           
  
        if self.hparams.limit_prompts is not None:
            all_data = all_data[: self.hparams.limit_prompts]
      
     
        print("number of training data\n",len(self.train_data))
        print("number of valiation data\n",len(self.val_data))
        print("number of test data\n",len(self.test_data))

      

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1)

class PromptDataPipe(MapDataPipe):
    def __init__(self, problems) -> None:
        super().__init__()
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):

        return self.problems[index]
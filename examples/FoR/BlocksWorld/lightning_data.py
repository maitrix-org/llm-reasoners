from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
import sys
from Executor import Executor
from utils import *
from bw_utils import *
sys.path.append("gpt-plan-benchmark/gpt_plan_test")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
import yaml
import json
from tarski.io import PDDLReader

def get_problem(instance, domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain)
    return reader.parse_instance(instance)

class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        args,
        tokenizer,
        train_size=0.2,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        with open('data/blocksworld/bw_config.yaml', 'r') as file:
            self.data = yaml.safe_load(file)
        self.prompts = json.load(open("data/blocksworld/my_mcts_prompts_update.json", 'r'))
        with open('data/blocksworld/bw_config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.domain_pddl = f'gpt-plan-benchmark/gpt_plan_test/instances/{self.config["domain_file"]}'
        self.tokenizer = tokenizer
        self.args = args
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        all_data = []
        train_data = json.load(open(f"./data/blocksworld/step_{self.args.step}.json", 'r'))
        train_data = train_data
        for d in train_data:
            problem = get_problem(d[0], self.domain_pddl)
            gt_plan_text = d[1]
            INIT, GOAL, PLAN = instance_to_text_blocksworld(problem, True, gt_plan_text, self.data)
            all_data.append([INIT, GOAL, PLAN])
        if self.hparams.limit_prompts is not None:
            all_data = all_data[: self.hparams.limit_prompts]
        self.train_data = PromptDataPipe(all_data[:15])
        self.val_data = PromptDataPipe(all_data[:15])
        self.test_data = PromptDataPipe(all_data[15:])

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

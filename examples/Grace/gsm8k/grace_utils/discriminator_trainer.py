from transformers import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker

class DiscriminatorMaxMarginTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._discriminator = self.model

    def compute_loss(self, model, inputs, return_outputs=False):
        pos_input_ids = inputs["pos_input_ids"]
        pos_attention_mask = inputs["pos_attention_mask"]
        pos_token_type_ids = inputs["pos_token_type_ids"]
        neg_input_ids = inputs["neg_input_ids"]
        neg_attention_mask = inputs["neg_attention_mask"]
        neg_token_type_ids = inputs["neg_token_type_ids"]

        pos_scores, neg_scores = model(pos_input_ids=pos_input_ids, 
                                       pos_attention_mask=pos_attention_mask, 
                                       pos_token_type_ids=pos_token_type_ids,
                                       neg_input_ids=neg_input_ids,
                                       neg_attention_mask=neg_attention_mask, 
                                       neg_token_type_ids=neg_token_type_ids)
        
        loss_type = getattr(self.args, 'loss_type', 'maxmargin')
        if loss_type == 'maxmargin':
            ## marginrankingloss
            criterion = nn.MarginRankingLoss(margin=self.args.margin, reduction='mean').to(pos_scores.device)
            loss = criterion(pos_scores, neg_scores, torch.ones_like(pos_scores).to(pos_scores.device))
        elif loss_type == 'logsigmoid':
            ## RLHF reward model ranking loss
            loss = -torch.log(torch.sigmoid((pos_scores - neg_scores))).mean()
        else:
            raise NotImplementedError("Loss type {} not implemented".format(loss_type))


        return (loss, pos_scores, neg_scores) if return_outputs else loss
    
    
    def prediction_step(self, model, inputs, prediction_loss_only=True,
                        ignore_keys=None):

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                if "loss" in self.args.dev_metric:
                    loss = self.compute_loss(model, inputs, return_outputs=False)
                elif "acc" in self.args.dev_metric:
                    assert self.args.greater_is_better 
                    pos_scores, neg_scores = self.compute_loss(model, inputs, return_outputs=True)[1:]
                    loss = (pos_scores > neg_scores).float().mean()
                else:
                    raise NotImplementedError("Only loss and acc are supported as dev metrics")
            loss = loss.mean().detach()

        return (loss, None, None)

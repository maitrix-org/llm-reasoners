import json, os 
import torch
from transformers import ElectraTokenizer, ElectraModel
import math

class ELECTRADiscriminator(torch.nn.Module):
    '''
    T5 discriminator is trained where the encoder takes in the question and a set of chain-of-thought prompts. The deocder is trained to in a sequence labeling kind where 0 represents an incorrect prefix and 1 represents a correct prefix.
    '''
    def __init__(self, model_name_or_path, args=None, device='cuda'):
        super().__init__()
        self.args = args
        self.model = ElectraModel.from_pretrained(model_name_or_path, is_decoder=True)
        #is_decoder is set to True to use only consider previous steps in the reasoning chain
        hidden_size = self.model.config.hidden_size
        self.out_linear = torch.nn.Linear(hidden_size, hidden_size)#.type(torch.bfloat16 if self.args.bf16 else torch.float32)
        self.classifier = torch.nn.Linear(hidden_size, 1)#.type(torch.bfloat16 if self.args.bf16 else torch.float32)
        self.loss_fct = torch.nn.CrossEntropyLoss()        
        self.device = device
        self.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids=None, shift_right=True):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        #import ipdb; ipdb.set_trace()
        hidden_out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        b, l, h = hidden_out.shape
        hidden_out = hidden_out.view(-1, h)
        linear_out = torch.relu(self.out_linear(hidden_out))
        logits = self.classifier(linear_out)
        logits = logits.view(b, l)
        return logits
    
class ELECTRAEnergyDiscriminator(torch.nn.Module):
    '''
    Disciminator takes a sequence of steps and produces and energy score for the whole sequence. Higher Energy means more likely to be correct.
    '''
    def __init__(self, model_name_or_path, args=None, device='cuda'):
        super().__init__()
        self.args = args
        self.model = ElectraModel.from_pretrained(model_name_or_path, is_decoder=False)
        #is_decoder is set to True to use only consider previous steps in the reasoning chain
        hidden_size = self.model.config.hidden_size
        self.out_linear1 = torch.nn.Linear(hidden_size, hidden_size)#.type(torch.bfloat16 if self.args.bf16 else torch.float32)
        self.out_linear2 = torch.nn.Linear(hidden_size, 1)#.type(torch.bfloat16 if self.args.bf16 else torch.float32)
        self.loss_fct = torch.nn.CrossEntropyLoss()        
        self.device = device
        self.to(device)

    def forward_scores(self, input_ids, attention_mask, token_type_ids=None):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        #import ipdb; ipdb.set_trace()
        hidden_out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        b, _, h = hidden_out.shape
        ## take representation of [CLS] token
        hidden_out = hidden_out[:, 0, :]
        hidden_out = hidden_out.view(b, h)
        linear_out = torch.relu(self.out_linear1(hidden_out))
        scores = self.out_linear2(linear_out)
        ## tanh 
        scores = torch.tanh(scores)
        return scores

    def forward(self, pos_input_ids, pos_attention_mask, pos_token_type_ids, neg_input_ids, neg_attention_mask, neg_token_type_ids):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        #import ipdb; ipdb.set_trace()
        pos_scores = self.forward_scores(pos_input_ids, pos_attention_mask, pos_token_type_ids)
        neg_scores = self.forward_scores(neg_input_ids, neg_attention_mask, neg_token_type_ids)
        return pos_scores, neg_scores
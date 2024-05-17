import json, os 
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class T5Discriminator(torch.nn.Module):
    
    '''
    T5 discriminator is trained where the encoder takes in the question and a set of chain-of-thought prompts. The deocder is trained to in a sequence labeling kind where 0 represents an incorrect prefix and 1 represents a correct prefix.
    '''
    def __init__(self, model_name_or_path, args=None, device='cuda'):
        super().__init__()
        self.args = args
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16 if self.args.bf16 else torch.float32)
        self.t5.lm_head.requires_grad_(False) # trick to make distributed training work
        self.out_linear = torch.nn.Linear(self.t5.config.d_model, self.t5.config.d_model // 2).type(torch.bfloat16 if self.args.bf16 else torch.float32)
        self.classifier = torch.nn.Linear(self.t5.config.d_model // 2, 1).type(torch.bfloat16 if self.args.bf16 else torch.float32)
        self.loss_fct = torch.nn.CrossEntropyLoss()

        train_n_layers = getattr(args, 'train_n_layers', -1)
        
        if train_n_layers > 0:
            nlayers = self.t5.config.num_layers
            print("training only the last {} layers of the discriminator".format(train_n_layers))
            ## freeze all layers in both encoder and decoder except the last n layers
            for i in range(nlayers - train_n_layers):
                for n, p in self.t5.named_parameters():
                    if 'block.{}'.format(i) in n:
                        p.requires_grad = False
        
        elif train_n_layers == 0:
            print("T5 discriminator is frozen and only the linear layers are trained")
            ## freeze all layers in both encoder and decoder
            for p in self.t5.parameters():
                p.requires_grad = False
            
        self.device = device
        self.to(device)

    def forward(self, enc_input_ids, dec_input_ids, enc_attn_mask, dec_attn_mask=None, disc_labels=None, shift_right=True):
        '''
        enc_input_ids: (batch_size, enc_seq_len)
        dec_input_ids: (batch_size, dec_seq_len)
        enc_attn_mask: (batch_size, enc_seq_len)
        dec_attn_mask: (batch_size, dec_seq_len)
        disc_labels: (batch_size, dec_seq_len)
        '''
        ## shift decoder input ids to the right        
        if shift_right:
            dec_input_ids = self.t5._shift_right(dec_input_ids)
        
        dec_attn_mask = (dec_input_ids != self.t5.config.pad_token_id).long()
        outputs = self.t5(input_ids=enc_input_ids, 
                    attention_mask=enc_attn_mask, 
                    decoder_input_ids=dec_input_ids,
                    decoder_attention_mask=dec_attn_mask,
                    output_hidden_states=True)
        
        ## get decoder last layer hidden states
        hidden_states = outputs.decoder_hidden_states[-1]
        #linear_out = gelu(self.out_linear(hidden_states))
        linear_out = torch.relu(self.out_linear(hidden_states))
        logits = self.classifier(linear_out).squeeze(-1)

        ## loss_mask 
        return logits, dec_attn_mask

    def forward_decoder_only(self, encoder_outputs, dec_input_ids, dec_attn_mask=None, shift_right=True):
        '''
        encoder_outputs: (batch_size, enc_seq_len, hidden_size)
        dec_input_ids: (batch_size, dec_seq_len)
        dec_attn_mask: (batch_size, dec_seq_len)
        '''
        ## shift decoder input ids to the right        
        if shift_right:
            dec_input_ids = self.t5._shift_right(dec_input_ids)
        
        dec_attn_mask = (dec_input_ids != self.t5.config.pad_token_id).long()
        outputs = self.t5.decoder(input_ids=dec_input_ids,
                    attention_mask=dec_attn_mask,
                    encoder_hidden_states=encoder_outputs,
                    output_hidden_states=True)
        
        ## get decoder last layer hidden states
        hidden_states = outputs.hidden_states[-1]
        linear_out = torch.relu(self.out_linear(hidden_states))
        logits = self.classifier(linear_out).squeeze(-1)

        ## loss_mask 
        return logits, dec_attn_mask

class T5EnergyDiscriminator(torch.nn.Module):
    '''
    Disciminator takes a sequence of steps and produces and energy score for the whole sequence. Higher Energy means more likely to be correct.
    IS comprised of a T5 encoder and a linear layer.
    '''
    def __init__(self, model_name_or_path, args=None, device='cuda'):
        super().__init__()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        #is_decoder is set to True to use only consider previous steps in the reasoning chain
        hidden_size = self.model.config.hidden_size
        self.out_linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.out_linear2 = torch.nn.Linear(hidden_size, 1)
        self.loss_fct = torch.nn.CrossEntropyLoss()        
        self.device = device
        self.to(device)
        self.pooling = args.pooling if hasattr(args, 'pooling') else 'max'

    def forward_scores(self, input_ids, attention_mask, token_type_ids=None):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        
        hidden_out = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        b, _, h = hidden_out.shape
        ## take max pooling over the sequence dimension over non-padded tokens
        
        if self.pooling == 'max':
            hidden_out = torch.where(attention_mask.unsqueeze(-1) == 0, torch.tensor(-1e9).to(self.device), hidden_out)
            hidden_out = torch.max(hidden_out, dim=1)[0] # (b, h)
        
        elif self.pooling == 'mean':
            hidden_out = torch.sum(hidden_out * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        elif self.pooling == 'cls':
            hidden_out = hidden_out[:, 0, :]

        
        hidden_out = hidden_out.view(b, h)
        linear_out = torch.relu(self.out_linear1(hidden_out))
        scores = self.out_linear2(linear_out)
        scores = torch.tanh(scores)
        return scores

    def forward(self, pos_input_ids, pos_attention_mask, 
                pos_token_type_ids, neg_input_ids, neg_attention_mask,         neg_token_type_ids):
        '''
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        '''
        pos_scores = self.forward_scores(pos_input_ids, pos_attention_mask, pos_token_type_ids)
        neg_scores = self.forward_scores(neg_input_ids, neg_attention_mask, neg_token_type_ids)
        return pos_scores, neg_scores
    

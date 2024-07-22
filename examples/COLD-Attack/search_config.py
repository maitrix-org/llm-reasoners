from reasoners import WorldModel, LanguageModel,Reasoner,SearchConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from task import *
from util import *
from opt_util import *
from typing import NamedTuple, List, Tuple, Dict, Any

class PromptState(NamedTuple):
    text: torch.tensor

class PromptAction(NamedTuple):
    new_prompt: str

class PromptSearchConfig_Paraphrase(SearchConfig[PromptState, PromptAction, str]):
    def __init__(self, world_model: AutoModelForCausalLM, 
                tokenizer,
                args
                ) -> None:
        # Example: 
        super().__init__()
        self.world_model = world_model
        self.tokenizer = tokenizer
        self.example = None
        self.args = args
        self.mask_t = None

    def update_example(self, example, prompt: dict = None) -> None:
        self.example = example
        x, x_mask, x_model_past, soft_forward_x, z_onehot, z_t, bad_words_t, z = example  
        self.x_model_past = x_model_past
        self.x_mask = x_mask
        self.soft_forward_x = soft_forward_x
        self.z_onehot = z_onehot
        self.z_t = z_t
        self.bad_words_t = bad_words_t
        self.x = x
        self.z = z
        x_seq = self.tokenizer.encode(x)[1:]
        x_seq_t = torch.tensor(x_seq, device=self.world_model.device, dtype=torch.long)
        self.x_seq_t = x_seq_t.unsqueeze(0).repeat(self.args.batch_size, 1)
        self.ref_embedding = get_ref_embedding(self.world_model, x, self.world_model.device, self.tokenizer)

    def get_actions(self, y_logits_: PromptState):
        # y_logits_ = y_logits + epsilon
        soft_forward_y = y_logits_ / 0.001

        if self.mask_t is None:
            soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        else:
            soft_forward_y = top_k_filter_3d(y_logits_, self.args.topk, mask=self.mask_t, extra_mask=self.x_mask, bad_mask=None) / 0.001

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y_logits_t = soft_forward(self.world_model, self.soft_forward_x, soft_forward_y, self.args.topk, extra_mask=self.x_mask, x_past=self.x_model_past, bad_mask=None) # without gradient

        _, indices_t = torch.topk(y_logits_t, self.args.topk)
        self.mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)
        flu_loss = soft_nll(
            top_k_filter_3d(y_logits_t / self.args.output_lgt_temp, self.args.topk, extra_mask=self.x_mask, bad_mask=None),
            y_logits_ / self.args.input_lgt_temp)


        soft_forward_y_ = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        if self.args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xyz_logits, xy_length = soft_forward_xyz(self.world_model, self.soft_forward_x, soft_forward_y_, self.z_onehot)
        else:
            xyz_logits, xy_length = soft_forward_xyz(self.world_model, self.soft_forward_x, soft_forward_y_, self.z_onehot)

        # Reshaping
        bz = self.args.batch_size
        lg = xyz_logits.shape[1]
        st = xy_length - 1
        ed = xyz_logits.shape[1] - 1
        xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
        z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

        c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
            z_logits,
            self.z_t.view(-1))
        c_loss_1 = c_loss_1.view(self.args.batch_size, -1).mean(-1)

        c_loss_2 = batch_log_bleulosscnn_ae(
            decoder_outputs=top_k_filter_3d(y_logits_, self.args.topk, mask=self.mask_t, extra_mask=self.x_mask).transpose(0, 1),
            target_idx=self.x_seq_t,
            ngram_list=list(range(1, self.args.counterfactual_max_ngram + 1))
        )
        # Paraphrasing loss2
        c_loss_3 = sim_score(self.world_model, top_k_filter_3d(y_logits_, self.args.topk, mask=self.mask_t, extra_mask=self.x_mask), self.ref_embedding)
        loss = self.args.goal_weight * c_loss_1 + self.args.rej_weight * c_loss_2  + self.args.lr_nll_portion * flu_loss - c_loss_3
        loss = loss.mean()
        # if iter < self.args.num_iters - 1: 
        #     loss.backward()
            # self.optim.step()
        
        return loss

    def reward(self, state: PromptState, action: PromptAction) -> Tuple[float, dict]:
        # return 0
        pass

    def decode(self, y_logits_: PromptState):
        
        text, _, last_text_ids = decode_with_model_topk(
        self.world_model, y_logits_, self.args.topk, self.soft_forward_x, self.x_model_past, self.tokenizer, extra_mask=None, bad_mask=None)
        text_post = text
        

        return text_post

class PromptSearchConfig_Suffix(SearchConfig[PromptState, PromptAction, str]):
    def __init__(self, world_model: AutoModelForCausalLM, 
                tokenizer,
                args
                ) -> None:
        # Example: 
        super().__init__()
        self.world_model = world_model
        self.tokenizer = tokenizer
        self.example = None
        self.args = args
        self.mask_t = None

    def update_example(self, example, prompt: dict = None) -> None:
        self.example = example
        x, x_mask, x_model_past, soft_forward_x, z_onehot, z_t, bad_words_t, z = example  
        self.x_model_past = x_model_past
        self.x_mask = x_mask
        self.soft_forward_x = soft_forward_x
        self.z_onehot = z_onehot
        self.z_t = z_t
        self.bad_words_t = bad_words_t
        self.x = x
        self.z = z

    def get_actions(self, y_logits_: PromptState):
        # y_logits_ = y_logits + epsilon
        soft_forward_y = y_logits_ / 0.001

        if self.mask_t is None:
            soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        else:
            soft_forward_y = top_k_filter_3d(y_logits_, self.args.topk, mask=self.mask_t, extra_mask=self.x_mask, bad_mask=None) / 0.001

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y_logits_t = soft_forward(self.world_model, self.soft_forward_x, soft_forward_y, self.args.topk, extra_mask=self.x_mask, x_past=self.x_model_past, bad_mask=None) # without gradient

        _, indices_t = torch.topk(y_logits_t, self.args.topk)
        self.mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)
        flu_loss = soft_nll(
            top_k_filter_3d(y_logits_t / self.args.output_lgt_temp, self.args.topk, extra_mask=self.x_mask, bad_mask=None),
            y_logits_ / self.args.input_lgt_temp)


        soft_forward_y_ = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        if self.args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xyz_logits, xy_length = soft_forward_xyz(self.world_model, self.soft_forward_x, soft_forward_y_, self.z_onehot)
        else:
            xyz_logits, xy_length = soft_forward_xyz(self.world_model, self.soft_forward_x, soft_forward_y_, self.z_onehot)

        # Reshaping
        bz = self.args.batch_size
        lg = xyz_logits.shape[1]
        st = xy_length - 1
        ed = xyz_logits.shape[1] - 1
        xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
        z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

        c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
            z_logits,
            self.z_t.view(-1))
        c_loss_1 = c_loss_1.view(self.args.batch_size, -1).mean(-1)

        c_loss_2 = batch_log_bleulosscnn_ae(
            decoder_outputs=y_logits_.transpose(0, 1),
            target_idx=self.bad_words_t,
            ngram_list=[1]
        )

        loss = self.args.goal_weight * c_loss_1 + 1 * flu_loss - self.args.rej_weight *  c_loss_2
        loss = loss.mean()
        # if iter < self.args.num_iters - 1: 
        #     loss.backward()
            # self.optim.step()
        
        return loss

    def reward(self, state: PromptState, action: PromptAction) -> Tuple[float, dict]:
        # return 0
        pass

    def decode(self, y_logits_: PromptState):
        
        text, _, last_text_ids = decode_with_model_topk(
        self.world_model, y_logits_, self.args.topk, self.soft_forward_x, self.x_model_past, self.tokenizer, extra_mask=None, bad_mask=None)
        text_post = text
        

        return text_post


class PromptSearchConfig_Position(SearchConfig[PromptState, PromptAction, str]):
    def __init__(self, world_model: AutoModelForCausalLM, 
                tokenizer,
                args
                ) -> None:
        # Example: 
        super().__init__()
        self.world_model = world_model
        self.tokenizer = tokenizer
        self.example = None
        self.args = args
        self.mask_t = None
        self.key_word = "environment"

    def update_example(self, example, prompt: dict = None) -> None:
        self.example = example
        x, x_mask, x_model_past, soft_forward_x, z_onehot, z_t, bad_words_t, z = example  
        self.x_model_past = x_model_past
        self.x_mask = x_mask
        self.soft_forward_x = soft_forward_x
        self.z_onehot = z_onehot
        self.z_t = z_t
        self.bad_words_t = bad_words_t
        self.x = x
        self.z = z

        x_ = self.tokenizer.encode(x)[1:]

        if self.args.control_type == "sentiment":
            self.control = "Write the output in an extremely exciting way. "
        elif self.args.control_type == "lexical":
            self.control = "The output written MUST include the following keywords: "
            keywords = self.key_word.split()
            for k in keywords:
                self.control += k
                self.control += ", "
            self.control = self.control.strip().strip(",")
            self.control += ". "
        elif self.args.control_type == "style":
            self.control = "Write the output as a Twitter post. "
        elif self.args.control_type == "format":
            self.control = "Write the output in a JSON format. "
        
        self.target_ = self.tokenizer.encode(z)[1:]  # delete the "." token we appended before
        self.target_t = torch.tensor(self.target_, device=self.world_model.device, dtype=torch.long)

        self.target_onehot = one_hot(self.target_t, dimension=self.tokenizer.vocab_size)
        self.target_onehot = self.target_onehot.repeat(self.args.batch_size, 1, 1)

        self.target_t = self.target_t.unsqueeze(0).repeat(self.args.batch_size, 1)

        length = self.args.length
        if length <= 0:
            length = self.target_t.shape[1] - length
        if self.args.verbose:
            print("x:\t|%s|\nz:\t|%s|\nlength:\t%d\ncontrol:\t%s" % (
                self.tokenizer.decode(x_), self.tokenizer.decode(self.target_), length, self.control))
        
        # target_mask: [batch_size, vocab_size]
        target_words = word_tokenize(z[:])  # delete the ". " token we appended before
        self.target_nonstop_words = [w.lower() for w in target_words if w.lower() not in stop_words and w.isalnum()]
        self.target_nonstop_words += [target_words[0]]  # add the first token
        self.target_nonstop_words = ' ' + ' '.join(self.target_nonstop_words)
        self.target_nonstop_ = self.tokenizer.encode(self.target_nonstop_words)
        print('|' + self.target_nonstop_words + '|')

        self.target_mask = np.zeros([self.tokenizer.vocab_size])
        self.target_mask[self.target_nonstop_] = 1.
        self.target_mask = torch.tensor(self.target_mask, device=self.world_model.device)
        self.target_mask = self.target_mask.unsqueeze(0).unsqueeze(0).repeat(self.args.batch_size, length, 1)

        self.control_ = self.tokenizer.encode(self.control)[1:]  # delete the "." token we appended before
        self.control_t = torch.tensor(self.control_, device=self.world_model.device, dtype=torch.long)
        
        self.control_onehot = one_hot(self.control_t, dimension=self.tokenizer.vocab_size)
        self.control_onehot = self.control_onehot.repeat(self.args.batch_size, 1, 1)
        self.control_t = self.control_t.unsqueeze(0).repeat(self.args.batch_size, 1)

    def get_actions(self, y_logits_: PromptState):
        # y_logits_ = y_logits + epsilon
        soft_forward_y = y_logits_ / 0.001

        if self.mask_t is None:
            soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        else:
            soft_forward_y = top_k_filter_3d(y_logits_, self.args.topk, mask=self.mask_t, extra_mask=self.x_mask, bad_mask=None) / 0.001

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y_logits_t = soft_forward(self.world_model, self.soft_forward_x, soft_forward_y, self.args.topk, extra_mask=self.x_mask, x_past=self.x_model_past, bad_mask=None) # without gradient

        _, indices_t = torch.topk(y_logits_t, self.args.topk)
        self.mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)
        flu_loss = soft_nll(
            top_k_filter_3d(y_logits_t / self.args.output_lgt_temp, self.args.topk, extra_mask=self.x_mask, bad_mask=None),
            y_logits_ / self.args.input_lgt_temp)


        soft_forward_y_ = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
        if self.args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xyz_target_logits, xyz_length = soft_forward_xyz_target(self.world_model, self.soft_forward_x, soft_forward_y_, self.control_onehot, self.target_onehot)
        else:
            xyz_target_logits, xyz_length = soft_forward_xyz_target(self.world_model, self.soft_forward_x, soft_forward_y_, self.control_onehot, self.target_onehot)

        # Reshaping
        bz = self.args.batch_size
        lg = xyz_target_logits.shape[1]
        st = xyz_length - 1
        ed = xyz_target_logits.shape[1] - 1
        xyz_target_logits = xyz_target_logits.view(-1, xyz_target_logits.shape[-1])
        target_logits = torch.cat([xyz_target_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)
        # print(target_logits.shape)
        c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
            target_logits,
            self.target_t.view(-1))
        c_loss_1 = c_loss_1.view(self.args.batch_size, -1).mean(-1)

        # future token loss
        if self.args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                xyz_logits, xy_length = soft_forward_xyz(self.world_model, self.soft_forward_x, soft_forward_y_, self.control_onehot)
        else:
            xyz_logits, xy_length = soft_forward_xyz(self.model, self.soft_forward_x, soft_forward_y_, self.control_onehot)

        bz = self.args.batch_size
        lg = xyz_logits.shape[1]
        st = xy_length - 1
        ed = xyz_logits.shape[1] - 1
        xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
        z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

        c_loss_2 = torch.nn.CrossEntropyLoss(reduction='none')(
            z_logits,
            self.control_t.view(-1))
        c_loss_2 = c_loss_2.view(self.args.batch_size, -1).mean(-1)

        c_loss_3 = batch_log_bleulosscnn_ae(
            decoder_outputs=y_logits_.transpose(0, 1),
            target_idx=self.bad_words_t,
            ngram_list=[1]
        )

        loss = self.args.goal_weight * c_loss_1 + 1.0 * flu_loss - self.args.rej_weight * c_loss_3 + 100 * c_loss_2
        loss = loss.mean()
        # if iter < self.args.num_iters - 1: 
        #     loss.backward()
            # self.optim.step()
        
        return loss

    def reward(self, state: PromptState, action: PromptAction) -> Tuple[float, dict]:
        # return 0
        pass

    def decode(self, y_logits_: PromptState):
        
        text, _, last_text_ids = decode_with_model_topk(
        self.world_model, y_logits_, self.args.topk, self.soft_forward_x, self.x_model_past, self.tokenizer, extra_mask=None, bad_mask=None)
        text_post = text
        

        return text_post

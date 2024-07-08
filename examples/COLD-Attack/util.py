import torch
import torch.nn.functional as F
import json
import os
import nltk
from nltk import tokenize
import torch
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('punkt')

import sys
import os
if os.path.isdir('/var/karen'):
    os.environ['TRANSFORMERS_CACHE'] = '/var/karen/workspace/Refinement-Generation/cache'
    sys.path.insert(0, '/var/karen/workspace/Refinement-Generation/')

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from difflib import SequenceMatcher

from bleuloss import batch_log_bleulosscnn_ae


def embed_inputs(embedding, logits, x_onehot=None, z_onehot=None, device='cuda'):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    # typically we embed a one-hot vector. But here since we work we work with dense representations,
    # we have softmax here to make sure that all the values of the input logits sum to one (similar to a 1-hot vector).
    probs = F.softmax(logits, dim=-1).type(torch.float16)
    # embedding : [vocab_size, embedding_size]
    # logits:     [batch_size, length, vocab_size]
    if x_onehot is not None:
        probs = torch.cat((x_onehot.type(torch.FloatTensor), probs.type(torch.FloatTensor)), dim=1)
    if z_onehot is not None:
        probs = torch.cat((probs.type(torch.FloatTensor), z_onehot.type(torch.FloatTensor)), dim=1)
        
    probs = probs.to(device)
    return torch.matmul(probs, embedding)

def embed_inputs_target(embedding, logits, x_onehot=None, z_onehot=None, target_onehot=None, device='cuda'):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    # typically we embed a one-hot vector. But here since we work we work with dense representations,
    # we have softmax here to make sure that all the values of the input logits sum to one (similar to a 1-hot vector).
    probs = F.softmax(logits, dim=-1).type(torch.float16)
    # embedding : [vocab_size, embedding_size]
    # logits:     [batch_size, length, vocab_size]
    if x_onehot is not None:
        probs = torch.cat((x_onehot.type(torch.FloatTensor), probs.type(torch.FloatTensor)), dim=1)
    if z_onehot is not None:
        probs = torch.cat((probs.type(torch.FloatTensor), z_onehot.type(torch.FloatTensor)), dim=1)
    if target_onehot is not None:
        probs = torch.cat((probs.type(torch.FloatTensor), target_onehot.type(torch.FloatTensor)), dim=1)
    probs = probs.to(device)
    return torch.matmul(probs, embedding)


def _greedy(logits):
    _, last = torch.topk(logits, k=1, dim=-1)
    return last


def top_k_filter_3d(logits, k, probs=False, mask=None, extra_mask=None, bad_mask=None):
    """
    logits.shape = [batch_size, length, vocab_size]
    extra_mask: [batch_size, length, vocab_size], 1 if reserve
    """
    BIG_CONST = 1e10
    if k == 0:
        return logits
    else:
        if mask is None:
            _, indices = torch.topk(logits, k)
            mask = torch.zeros_like(logits).scatter_(2, indices, 1)
        if bad_mask is not None:
            mask = torch.mul(mask, bad_mask)
        if extra_mask is not None:
            mask = ((mask + extra_mask) > 0).float()
        if probs:
            return logits * mask
        return logits * mask + -BIG_CONST * (1-mask)


def top_k_filter(logits, k, probs=False):
    BIG_CONST = 1e10
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def _topk(logits, k=10):
    logits = top_k_filter(logits, k)
    probs = F.softmax(logits, dim=-1)
    last = torch.multinomial(probs, num_samples=1)
    return last

def top_p(logits, thres = 0.5, filter_value=-float('Inf')):
    assert len(logits.shape) == 1
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > thres

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits


def get_text_from_logits(logits, tokenizer):
    output_so_far = None
    last = None
    logp = 0
    for i in range(logits.shape[1]):
        last = _greedy(logits[:, i, :])
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        logp += logits[:, i, :].log_softmax(-1).data.cpu().numpy()[:, last.data.cpu().numpy()]

    nll = -logp
    batch_size = output_so_far.shape[0]
    text = []
    for i in range(batch_size):
        text_i = tokenizer.decode(output_so_far[i].tolist())
        text_i = text_i.replace('\n', ' ')
        text_i += ". "
        text.append(text_i)

    return text, nll, output_so_far


def get_text_from_logits_topk(logits, tokenizer, top_k=1):
    output_so_far = None
    last = None
    logp = 0

    for i in range(logits.shape[1]):
        last = _topk(logits[:, i, :], top_k)
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        logp += logits[:, i, :].log_softmax(-1)[:, last.item()].item()

    nll = -logp
    text = tokenizer.decode(output_so_far.tolist()[0])
    text = text.replace('\n', ' ')
    return text, nll, output_so_far



def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1), 1)
    onehot.to(tensor.device)
    return onehot


def initialize(model, x, length, temperature, batch_size, device, tokenizer):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # print(x.shape)
    # past = None
    # last_token_embedding = None
    # logits_so_far = None
    # for i in range(length):
    #     # for the first iteration, `past` is None
    #     if past is None:
    #         x_last_token = x[:, -1:]
    #         last_token_embedding = model.get_input_embeddings()(x_last_token)

    #         # if the input length is longer than a single token
    #         if x.shape[1] > 1:
    #             x_except_last_token = x[:, :-1]
    #             model_outputs = model(x_except_last_token)
    #             past = model_outputs.past_key_values
        
    #     model_outputs = model(past_key_values=past, inputs_embeds=last_token_embedding)
    #     logits = model_outputs.logits
    #     # print(logits.shape)
    #     past = model_outputs.past_key_values

    #     logits = logits[:, -1, :] / temperature
    #     logits = logits.unsqueeze(1)
    #     logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)

    # output = model.generate(x, max_length=length + x.shape[-1])
    # logits = model(output).logits

    output = model.generate(x, max_length=length + x.shape[-1], do_sample=True, top_k=10)
    logits = model(output).logits
    logits_so_far = logits[:, -(length+1):-1, :] / temperature
    # print(logits_so_far.shape)

    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
    return logits_so_far


def decode_with_model_topk(model, y_logits, topk, x_onehot, x_past, tokenizer, extra_mask=None, bad_mask=None):
    # y_logits : [bsz, length, vocab_size]
    # x_onehot : [bsz, 1     , vocab_size]
    # extra_mask:[bsz, length, vocab_size]
    assert x_onehot.shape[1] == 1, x_onehot.shape
    length = y_logits.shape[1]
    past = x_past
    embeddings_weight =  model.get_input_embeddings().weight
    input_embeds = torch.matmul(x_onehot.float().to(embeddings_weight.dtype), embeddings_weight)
    mask_t_all = None
    logits_so_far = None
    # print(y_logits.shape)
    # print(x_onehot.shape)
    # print(x_past)
    for i in range(length):
        model_outputs = model(past_key_values=past, inputs_embeds=input_embeds, use_cache=True) #送进去历史信息
        past = model_outputs.past_key_values    
        logits_t = model_outputs.logits[:, -1:, :]  #选出来最后一个单词的logits
        assert logits_t.shape[1] == 1, logits_t.shape   
        _, indices_t = torch.topk(logits_t, topk)   #最后一个单词的topk logits
        mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)   #变mask
        if bad_mask != None:
            mask_t = torch.mul(mask_t, bad_mask)
        mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)   #第i个单词的topk-mask
        logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)  # 生成的y的logits
        if i < length - 1:
            if extra_mask is None:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
            else:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
            input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1).to(embeddings_weight.dtype), embeddings_weight)   # 得到第i个单词的embedding
    return get_text_from_logits(
        top_k_filter_3d(y_logits, topk, mask=mask_t_all, extra_mask=extra_mask),
        tokenizer)

def soft_forward_loss(model, y_logits, topk, x_onehot, x_past, extra_mask=None, bad_mask=None):
    # y_logits : [bsz, length, vocab_size]
    # x_onehot : [bsz, 1     , vocab_size]
    # extra_mask:[bsz, length, vocab_size]
    xy_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        device=x_onehot.device
    )
    # print(x_past.shape)
    # embed_inputs: [bsz, length, embed_size]
    xy_logits = model(past_key_values=x_past, inputs_embeds=xy_embeds).logits
    x_length = x_onehot.shape[1]
    y_logits = xy_logits[:, x_length - 1:-1, :]

    # past = x_past

    # # 初始化输入的嵌入表示
    # input_embeds = embed_inputs(
    #     model.get_input_embeddings().weight,
    #     y_logits,
    #     x_onehot=x_onehot,
    #     device=x_onehot.device
    # )

    # # 获取输入文本的长度
    # x_length = x_onehot.shape[1]

    # # 初始化y_logits
    # y_logits = None

    # # 循环生成y_logits
    # for i in range(x_length - 1, input_embeds.shape[1]):
    #     # 从历史信息和输入嵌入计算模型的输出
    #     model_outputs = model(past_key_values=past, inputs_embeds=input_embeds[:, i:i+1, :])
        
    #     # 更新历史信息
    #     past = model_outputs.past_key_values
        
    #     # 提取当前时间步骤的logits
    #     logits_t = model_outputs.logits[:, -1:, :]
        
    #     # 将logits_t追加到y_logits
    #     if y_logits is None:
    #         y_logits = logits_t
    #     else:
    #         y_logits = torch.cat((y_logits, logits_t), dim=1)
    _, indices_t = torch.topk(y_logits, topk)   #最后一个单词的topk logits
    mask_t_all = torch.zeros_like(y_logits).scatter_(2, indices_t, 1)

    logp = F.log_softmax(y_logits, dim=-1)
    p = mask_t_all

    return -(p * logp).sum(dim=-1).mean(dim=-1)

    # mask_t_all = None
    # logits_so_far = None
    # # print(y_logits.shape)
    # # print(x_onehot.shape)
    # # print(x_past)
    # length = y_logits.shape[1]
    # for i in range(length):
    #     model_outputs = model(past_key_values=past, inputs_embeds=input_embeds) #送进去历史信息
    #     past = model_outputs.past_key_values    
    #     logits_t = model_outputs.logits[:, -1:, :]  #选出来最后一个单词的logits
    #     assert logits_t.shape[1] == 1, logits_t.shape   
    #     _, indices_t = torch.topk(logits_t, topk)   #最后一个单词的topk logits
    #     mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)   #变mask
    #     mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)   #第i个单词的topk-mask
    #     logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)  # 生成的y的logits
    #     if i < length - 1:
    #         if extra_mask is None:
    #             y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
    #         else:
    #             y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
    #         input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1), model.get_input_embeddings().weight)   # 得到第i个单词的embedding

    # _, indices_t = torch.topk(logits_so_far, topk)   #最后一个单词的topk logits
    # mask_t_all = torch.zeros_like(logits_so_far).scatter_(2, indices_t, 1)

    # logp = F.log_softmax(logits_so_far, dim=-1)
    # p = mask_t_all

    # return -(p * logp).sum(dim=-1).mean(dim=-1)
    

def soft_backward_loss(model, y_logits_, yz_logits_rev, topk):
    # y_logits_rev: 
    embeddings_weight = model.get_input_embeddings().weight[1:yz_logits_rev.shape[-1]+1]
    y_embeds = embed_inputs(
        embeddings_weight,
        yz_logits_rev,
        device=yz_logits_rev.device
    )
    y_logits = model(inputs_embeds=y_embeds).logits
    yz_logits_rev_rev_t = torch.flip(y_logits, [1])                      # 再翻转回来
    yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_.shape[-1] + 1]   
    yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_.shape[1], :]
    
    tmp_logits = yz_logits_rev_rev_t_
    repetition_mask = torch.cat([F.softmax(tmp_logits[:, 1:, :], dim=-1),
                                 torch.zeros_like(tmp_logits[:, -1:, :])], dim=1)
    yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_ - repetition_mask * 1e4
    # yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_.detach()

    _, indices_t = torch.topk(yz_logits_rev_rev_t_, topk)   #最后一个单词的topk logits
    mask_t_all = torch.zeros_like(yz_logits_rev_rev_t_).scatter_(2, indices_t, 1)   #变mask

    logp = F.log_softmax(yz_logits_rev_rev_t_, dim=-1)

    p = mask_t_all
    
    return -(p * logp).sum(dim=-1).mean(dim=-1)

def post_process(text_ids, model, max_length, length, tokenizer, device):
    # sentence completion
    text_ids_complete = sentence_completion(text_ids, model, max_length, device)
    batch_size = text_ids.shape[0]
    text_so_far_all = []
    for bi in range(batch_size):
        text_complete = tokenizer.decode(text_ids_complete[bi].tolist())
        text_complete = text_complete.replace('\n', ' ')

        # truncate to minimal complete text
        sents = nltk.sent_tokenize(text_complete)
        text_so_far = None
        length_so_far = 0
        for i, sent in enumerate(sents):
            text_so_far = sent if text_so_far is None else text_so_far + ' ' + sent
            sent_length = len(sent.split())
            length_so_far += sent_length
            if length_so_far >= length:
                break
        text_so_far += ". "
        text_so_far_all.append(text_so_far)
    return text_so_far_all


def sentence_completion(text_ids, model, max_length, device):
    output_so_far = text_ids
    past = None
    last_embeds = None
    # logits_so_far = None
    for i in range(max_length - text_ids.shape[1]):
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            last_embeds = model.get_input_embeddings()(last)

            if output_so_far.shape[1] > 1:
                model_outputs = model(output_so_far[:, :-1])
                past = model_outputs.past_key_values

        model_outputs = model(past_key_values=past, inputs_embeds=last_embeds, use_cache=True)
        logits = model_outputs.logits
        past = model_outputs.past_key_values

        last = _greedy(logits[:, -1, :])
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        # last_embeds = get_input_embeds(model.get_input_embeddings(), logits[:, -1:, :], device=device)
        last_embeds = model.get_input_embeddings()(last)

    # output = model.generate(text_ids, max_length=max_length)
    # logits = model(output).logits
    # output_so_far = None
    # for i in range(max_length):
    #     last = _greedy(logits[:, i, :])
    #     output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)


    return output_so_far


def soft_distance(logits_perturbed, logits):
    return torch.nn.MSELoss()(logits_perturbed, logits)


def soft_nll(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)


def soft_nll_detach(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1).detach()
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean()


def additional_nll(logits, cur_text_ids):
    return torch.nn.CrossEntropyLoss()(
        logits.view(-1, logits.shape[-1]),
        cur_text_ids.view(-1)
    )


def soft_forward(model, x_onehot, y_logits, topk, extra_mask=None, x_past=None, detach=True, bad_mask=None):
    '''
    computes logits for $y$, based on a fixed context $y$ and the current logit distribution of $y$
    :param model:
    :param x_onehot:
    :param y_logits:
    :return:
    '''
    xy_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        device=y_logits.device
    )
    # embed_inputs: [bsz, length, embed_size]
    xy_logits = model(past_key_values=x_past, inputs_embeds=xy_embeds, use_cache=True).logits
    # print(xy_logits.shape)
    if x_onehot != None:
        x_length = x_onehot.shape[1]
        y_logits = xy_logits[:, x_length - 1:-1, :]
    else:
        x_length = 1
        y_logits = xy_logits
    if detach:
        return y_logits.detach()
    else:
        return y_logits
    length = y_logits.shape[1]
    past = x_past
    input_embeds = torch.matmul(x_onehot.float(), model.get_input_embeddings().weight)
    

    mask_t_all = None
    logits_so_far = None
    length = y_logits.shape[1]
    for i in range(length):
        model_outputs = model(past_key_values=past, inputs_embeds=input_embeds) #送进去历史信息
        
        # past = model_outputs.past_key_values    
        logits_t = model_outputs.logits[:, -1:, :]  #选出来最后一个单词的logits
        assert logits_t.shape[1] == 1, logits_t.shape   
        _, indices_t = torch.topk(logits_t, topk)   #最后一个单词的topk logits
        mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)   #变mask
        if bad_mask != None:
            mask_t = torch.mul(mask_t, bad_mask)
        mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)   #第i个单词的topk-mask
        logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)  # 生成的y的logits
        if i < length - 1:
            if extra_mask is None:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
            else:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
            input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1), model.get_input_embeddings().weight)   # 得到第i个单词的embedding
        # memory_difference = end_memory - start_memory
        # print(f"Memory Difference: {end_memory / (1024 ** 2)} MB")
    if detach:
        return logits_so_far.detach()
    else:
        return logits_so_far



def soft_forward_xyz(model, x_onehot, y_logits, z_onehot):
    '''
    computes logits for $y$, based on a fixed context $y$ and the current logit distribution of $y$
    :param model:
    :param x_onehot:
    :param y_logits:
    :return:
    '''
    xyz_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        z_onehot=z_onehot,
        device=y_logits.device
    )
    xyz_logits = model(inputs_embeds=xyz_embeds).logits
    if x_onehot is not None:
        xy_length = x_onehot.shape[1] + y_logits.shape[1]
    else:
        xy_length = y_logits.shape[1]
    return xyz_logits, xy_length

def soft_forward_xyz_target(model, x_onehot, y_logits, z_onehot, target_onehot):
    '''
    computes logits for $y$, based on a fixed context $y$ and the current logit distribution of $y$
    :param model:
    :param x_onehot:
    :param y_logits:
    :return:
    '''

    xyz_target_embeds = embed_inputs_target(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        z_onehot=z_onehot,
        target_onehot=target_onehot,
        device=y_logits.device
    )
    xyz_target_logits = model(inputs_embeds=xyz_target_embeds).logits
    if x_onehot is not None:
        xyz_length = x_onehot.shape[1] + y_logits.shape[1] + z_onehot.shape[1]
    else:
        xyz_length = y_logits.shape[1]
    return xyz_target_logits, xyz_length

def soft_backward(model, y_logits_rev):
    embeddings_weight = model.get_input_embeddings().weight[1:y_logits_rev.shape[-1]+1]
    y_embeds = embed_inputs(
        embeddings_weight,
        y_logits_rev,
        device=y_logits_rev.device
    )
    y_logits_ = model(inputs_embeds=y_embeds).logits
    return y_logits_[:, :-1, :]


def soft_backward_steps(model, y_logits):
    device = y_logits.device
    past = None
    last_embeds = None
    logits_so_far = None
    for i in range(y_logits.shape[1]-2, -1, -1):
        last = y_logits[:, i:i+1]
        last_embeds = embed_inputs(model.get_input_embeddings(), last, device=device)

        model_outputs = model(past_key_values=past, inputs_embeds=last_embeds, use_cache=True)
        past = model_outputs.past_key_values

        logits = model_outputs.logits
        logits = logits[:, -1, :]
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)

    return logits_so_far



def constraint_loss(logits, cs_onehot, cs_ids):
    """
    constraint loss with mask
    cs_ids: [batch_size, num_cs]
    """
    log_ps = logits.log_softmax(-1).unsqueeze(2)  # shape: [batch_size, length, 1, vocab_size]
    constraint_max_log_ps_ = (log_ps * cs_onehot.unsqueeze(1)).max(1)[0].sum(-1)  # shape: [batch_size, num_cs]

    log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
    cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_cs, length]
    mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_cs]
    mask = (mask < 1).type(torch.FloatTensor)
    mask = mask.to(constraint_max_log_ps_.device)

    loss = - (constraint_max_log_ps_ * mask).sum()

    if mask.sum() != 0:
        loss = loss / mask.sum()
    else:
        loss = 0

    return loss


def constraint_loss_with_variants(logits, cs_onehot_all, cs_ids_all):
    """
    constraint loss with mask
    cs_ids_all: list of tensor [batch_size, num_variants], of length num_cs
    """
    device = logits.device
    log_ps = logits.log_softmax(-1).unsqueeze(2)  # shape: [batch_size, length, 1, vocab_size]

    num_cs = len(cs_onehot_all)
    loss_all = 0
    mask_sum = 0
    for i in range(num_cs):
        cs_onehot = cs_onehot_all[i]
        cs_ids = cs_ids_all[i]
        constraint_max_log_ps_ = (log_ps * cs_onehot.unsqueeze(1)).max(1)[0].sum(-1)  # shape: [batch_size, num_variants]

        log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
        cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_variants, length]
        mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_variants]
        #mask = (mask >= 1).type(torch.FloatTensor)
        mask = (mask.sum(1) < 1).type(torch.FloatTensor)  # shape: [batch_size]. mask = 0 if any of the variants already occurs
        mask = mask.to(device)

        loss_i = - (constraint_max_log_ps_.max(1)[0] * mask).mean()  # average over batch_size

        loss_all += loss_i
        mask_sum += mask

    if mask_sum != 0:
        loss_all = loss_all / mask_sum

    return loss_all #, mask_sum


def constraint_loss_with_variants_by_ppl(logits, cs_onehot_all, cs_ids_all, probs_t):
    device = logits.device
    batch_size = logits.shape[0]
    log_ps = logits.log_softmax(-1).unsqueeze(2)
    ps_t = probs_t.unsqueeze(2)

    num_cs = len(cs_onehot_all)
    loss_all = 0
    mask_sum = 0
    for i in range(num_cs):
        cs_onehot = cs_onehot_all[i]
        cs_ids = cs_ids_all[i]

        cs_onehot_ = cs_onehot.unsqueeze(1).type(torch.FloatTensor).to(device)
        cs_onehot_ = cs_onehot_.repeat(batch_size, 1, 1, 1).type(torch.FloatTensor).to(device)
        ppl_max_idx = (ps_t * cs_onehot_).argmax(1)  # [batch_size, num_variants, vocab_size]
        ppl_max_idx_onehot = torch.zeros_like(log_ps * cs_onehot_).scatter_(1, ppl_max_idx.unsqueeze(1), cs_onehot_)

        constraint_max_log_ps_ = (log_ps * ppl_max_idx_onehot).sum(1).sum(-1)  # shape: [batch_size, num_variants]

        ## Mask
        log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
        cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_variants, length]
        mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_variants]
        mask = (mask.sum(1) < 1).type(torch.FloatTensor)  # shape: [batch_size]. mask = 0 if any of the variants already occurs
        mask = mask.to(device)

        loss_i = - constraint_max_log_ps_.max(1)[0] * mask

        loss_all += loss_i  # shape: [batch_size]
        mask_sum += mask  # shape: [batch_size]

    loss_all = loss_all / (mask_sum + 1e-8)

    return loss_all


def constraint_loss_by_ppl(logits, cs_onehot, cs_ids, logits_t):
    device = logits.device
    log_ps = logits.log_softmax(-1).unsqueeze(2)

    cs_onehot_ = cs_onehot.unsqueeze(1).type(torch.FloatTensor).to(device)
    ps_t = logits_t.softmax(-1).unsqueeze(2)
    ppl_max_idx = (ps_t * cs_onehot_).argmax(1)  # [batch_size, num_cs, vocab_size]
    ppl_max_idx_onehot = torch.zeros_like(log_ps * cs_onehot_).scatter_(1, ppl_max_idx.unsqueeze(1), cs_onehot_)

    constraint_max_log_ps_ = (log_ps * ppl_max_idx_onehot).sum(1).sum(-1)  # shape: [batch_size, num_cs]

    ## Mask
    log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
    cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_cs, length]
    mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_cs]
    mask = (mask < 1).type(torch.FloatTensor)
    mask = mask.to(device)

    loss = - (constraint_max_log_ps_ * mask).sum()

    if mask.sum() != 0:
        loss = loss / mask.sum()
    else:
        loss = 0

    return loss


def constraint_loss_all(logits, cs_onehot, cs_ids):
    device = logits.device

    log_ps = logits.log_softmax(-1).unsqueeze(2)
    constraint_max_log_ps_ = (log_ps * cs_onehot.unsqueeze(1)).mean(1).sum(-1)  # shape: [batch_size, num_cs]

    ## Mask
    log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
    cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_cs, length]
    mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_cs]
    mask = (mask < 1).type(torch.FloatTensor)
    mask = mask.to(device)

    loss = - (constraint_max_log_ps_ * mask).sum()

    if mask.sum() != 0:
        loss = loss / mask.sum()
    else:
        loss = 0

    return loss

def _constraint_loss2(logits, cs_onehot):
    '''
    a re-implementation of `_constraint_loss` with a slightly different logic.
    TODO: keep only one of these functions
    '''
    logits = logits.squeeze(0) # drop the empty dimension
    cs_onehot = cs_onehot.float().squeeze(0) # drop the empty dimension and change into float (since torch matrix multiplication does not support integers)
    cs_onehot = torch.transpose(cs_onehot, 0, 1)
    selected_logits = torch.matmul(logits, cs_onehot) # dim: length x # of constraints
    max_logits_per_constraint, _ = selected_logits.max(0) # select the highest logits for each constraint
    loss = - max_logits_per_constraint.sum() / selected_logits.size(1)
    return loss

def print_topk_stats(logits, tokenizer):
    logits_lg, topk_index_y = torch.topk(F.softmax(logits[0, :3, :], dim=-1), 3)
    print(logits_lg.data.cpu().numpy())
    print(topk_index_y.data.cpu().numpy())
    lgs = [int(x[0]) for x in topk_index_y.data.cpu().numpy()]
    for a in lgs:
        print('|', tokenizer.decode(a), '| ', end='', flush=True)
    print()
    print("===============================")
    return topk_index_y

def pre_filter(model, y_logits, topk, x_onehot, x_past, tokenizer, extra_mask=None):
    # y_logits : [bsz, length, vocab_size]
    # x_onehot : [bsz, 1     , vocab_size]
    # extra_mask:[bsz, length, vocab_size]
    assert x_onehot.shape[1] == 1, x_onehot.shape
    length = y_logits.shape[1]
    past = x_past
    input_embeds = torch.matmul(x_onehot.float(), model.get_input_embeddings().weight)
    mask_t_all = None
    logits_so_far = None
    # print(y_logits.shape)
    # print(x_onehot.shape)
    # print(x_past)
    for i in range(length):
        model_outputs = model(past_key_values=past, inputs_embeds=input_embeds)
        past = model_outputs.past_key_values
        logits_t = model_outputs.logits[:, -1:, :]
        assert logits_t.shape[1] == 1, logits_t.shape
        _, indices_t = torch.topk(logits_t, topk)
        mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)
        mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)
        logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)
        if i < length - 1:
            if extra_mask is None:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
            else:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
            input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1), model.get_input_embeddings().weight)
    return get_text_from_logits(
        top_k_filter_3d(y_logits, topk, mask=mask_t_all, extra_mask=extra_mask),
        tokenizer)

def collect_json_lines(model_output_json_file):
    with open(model_output_json_file, 'r') as fr:
        lines = fr.readlines()
        json_lines = [json.loads(x.strip()) for x in lines]
        return json_lines

def post_sent(text_complete):
    sents = nltk.sent_tokenize(text_complete)
    sent = ' '.join(sents[0].strip().split())
    return sent
    # return sents[0]

def _has_repeat_sent(hyp):
    """
    Detect if the sentences in `hyp` are repeat.
    Args:
        hyp: A list of three sentences.
    """
    if len(hyp) <= 1:
        return False

    for i in range(1, len(hyp)):
        a = hyp[i-1]
        b = hyp[i]

        if a == b:
            return True

        s = SequenceMatcher(None, a, b)
        if len(a) > 5 and len(b) > 5 and s.ratio() >= 0.85:
            return True

    return False


def _has_repeat_substring(s, MINLEN=4, MINCNT=4):
    d = {}
    has_repeat = False
    for sublen in range(int(len(s)/MINCNT)-1, MINLEN-1, -1):
        for i in range(0, len(s)-sublen):
            sub = s[i:i+sublen]
            if len(sub.strip()) < sublen:
                continue
            cnt = s.count(sub)
            if cnt >= MINCNT and sub not in d:
                 d[sub] = cnt
                 # print('repeat_substring: |' + sub + '| in |' + s + '|')
                 has_repeat = True
                 break
        if has_repeat:
            break
    return has_repeat


def has_repeat(sents_for_substr):
    """
    Detect if the hypothesis text has repeat patterns.
    """
    has_repeat_substring = False
    for h in sents_for_substr:
        has_repeat_substring = has_repeat_substring or _has_repeat_substring(h) or _has_repeat_substring(h, MINLEN=20, MINCNT=2)
    # print(has_repeat_substring)
    # print(_has_repeat_sent(hyp))
    return has_repeat_substring


def write_json_lines(json_lines, fout, model, tokenizer, device):
    with open(fout, 'w') as fw:
        for line in json_lines:
            input_text = line['generation_complete'][0][0]
            # input_text = line['counterfactual']

            ori_ending = line['original_ending']
            ori_endings = tokenize.sent_tokenize(ori_ending)
            z = ori_endings[0].strip()

            gens = line['generation_complete'][0][1]
            proc_gens = [post_sent(x) for x in gens]
            pg_dict, gens_ranked, pg_dict_top, gens_ranked_top = process_batching_counterfactual_outputs(
                proc_gens, input_text, z, model, tokenizer, device)
            line['proced'] = proc_gens
            line['ppl_gens'] = pg_dict
            line['gens_ranked'] = gens_ranked
            line['ppl_gens_top'] = pg_dict_top
            line['gens_ranked_top'] = gens_ranked_top
            # print(line)
            # exit()
            fw.write(json.dumps(line) + '\n')


def compute_ppl_line(model, tokenizer, line):
    line = line.strip()
    #print(line)
    line_ = tokenizer.encode(line)
    line_t = torch.tensor(line_, dtype=torch.long).cuda()
    loss = model(input_ids=line_t, labels=line_t).loss
    loss = loss.detach().clone().data.cpu().numpy()
    ppl = np.exp(loss)
    return ppl


def compute_loss(model, tokenizer, device, x="", z="", y="", constraints=None, args=None, model_back=None, zz=None):
    '''
    x: left context   (prompt in lexical constrained task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''
    batch_size = 2

    x_ = tokenizer.encode(x)
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(batch_size, 1)
    x_onehot = x_onehot.repeat(batch_size, 1, 1)

    z_ = tokenizer.encode(z)[1:] # delete the "." token we appended before
    z_t = torch.tensor(z_, device=device, dtype=torch.long)
    z_t = z_t.unsqueeze(0).repeat(batch_size, 1)

    y_ = tokenizer.encode(y)[1:] # delete the "." token we appended before
    y_t = torch.tensor(y_, device=device, dtype=torch.long)
    y_onehot = one_hot(y_t, dimension=tokenizer.vocab_size)
    y_onehot = y_onehot.repeat(batch_size, 1, 1)
    y_t = y_t.unsqueeze(0).repeat(batch_size, 1)

    y_logits_ = y_onehot / 0.0001

    c_loss = batch_log_bleulosscnn_ae(
        decoder_outputs=y_logits_.transpose(0, 1),
        target_idx=z_t,
        ngram_list=[2, 3]
    )

    return c_loss.mean().item()


def rank_and_filter(candidates, input_text, z, model, tokenizer, device, no_loss_rerank):

    # de-duplicate
    candidates = list(dict.fromkeys(candidates))

    ppl_list = []
    ppl_y_list = []
    loss_list = []
    for line in candidates:
        line = line.strip()
        y = ' '.join(line.split())
        # y = line
        xy = input_text + ' ' + line
        # print(xy)
        # exit()
        x_sents = nltk.sent_tokenize(input_text)
        if has_repeat(sents_for_substr=[y], sents_for_sent=x_sents+[y]) or len(tokenizer.encode(y)) <= 4:
            ppl_list.append(10000.0)
            ppl_y_list.append(10000.0)
            loss_list.append(10000.0)
        else:
            ppl = compute_ppl_line(model, tokenizer, device, xy)
            ppl_list.append(round(ppl, 2))

            ppl_y = compute_ppl_line(model, tokenizer, device, y)
            ppl_y_list.append(round(ppl_y, 2))

            loss = compute_loss(model, tokenizer, device,
                                x=input_text, z=". " + z, y=". " + y)
            loss_list.append(loss)

    sort_index = sorted(range(len(ppl_list)), key=lambda k: ppl_list[k])
    ppls_reorder = [ppl_list[i] for i in sort_index]
    ppls_y_reorder = [ppl_y_list[i] for i in sort_index]
    loss_reorder = [loss_list[i] for i in sort_index]
    gens_complete_reorder = [candidates[i] for i in sort_index]

    pg_dict = []
    for p, py, l, g in zip(ppls_reorder, ppls_y_reorder, loss_reorder, gens_complete_reorder):
        pg_dict.append({"ppl": str(p), "ppl_y": str(py), "loss": str(l), "gen": g})

    if len(ppls_reorder) <= 1:
        sort_len = 1
    elif ppls_reorder[1]-ppls_reorder[0] > 10:
        sort_len = 1
    elif len(ppls_reorder) <= 2:
        sort_len = 1
    elif ppls_reorder[2]-ppls_reorder[0] > 10:
        sort_len = 2
    else:
        sort_len = 3

    if no_loss_rerank:
        return gens_complete_reorder[0]

    sort_index = sorted(range(sort_len), key=lambda k: loss_reorder[k])
    sort_index = sort_index
    ppls_reorder_top = [ppls_reorder[i] for i in sort_index]
    ppls_y_reorder_top = [ppls_y_reorder[i] for i in sort_index]
    loss_reorder_top = [loss_reorder[i] for i in sort_index]
    gens_complete_reorder_top = [gens_complete_reorder[i] for i in sort_index]

    pg_dict_top = []
    for p, py, l, g in zip(ppls_reorder_top, ppls_y_reorder_top, loss_reorder_top, gens_complete_reorder_top):
        pg_dict_top.append({"ppl": str(p), "ppl_y": str(py), "loss": str(l), "gen": g})

    return gens_complete_reorder_top[0]

def _get_adverbs_and_nnps(z_words):
    pos = nltk.pos_tag(z_words)
    adverbs = [w[0] for w in pos if 'RB' in w[1]]
    nnps = [w[0] for w in pos if 'NNP' in w[1]]
    return adverbs, nnps

def _get_keywords(z, x, args):
    stop_words = set(stopwords.words('english'))
    z_words = word_tokenize(z)
    z_adverbs, z_nnps = _get_adverbs_and_nnps(z_words)
    ret_words = []
    for w in z_words:
        if w in z_nnps:
            if w not in ret_words:
                ret_words.append(w)
        else:
            w = w.lower()
            if w not in stop_words and w.isalnum() and w not in z_adverbs and w not in ret_words:
                ret_words.append(w)

    if args.abductive_filterx:
        x_words = word_tokenize(x)
        ret_words = [w for w in ret_words if w not in x_words]

    return ' '.join(ret_words)


def calculate_coverage(output_ln, key_words):
    x_words = word_tokenize(x)
    x_words = set(x_words)
    count = len(x_words.intersection(key_words))
    ratio = count / len(key_words)
    return ratio

def get_gpt_ppl(text_list, gpt_model, gpt_tokenizer, device):
    # gpt_model = GPT2LMHeadModel.from_pretrained("./models/gpt2-medium").to(device)
    # gpt_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2-")
    gpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    ppl_list = []
    
    for line in text_list:
        text = line

        encoded_input = gpt_tokenizer(text, padding=True, return_tensors='pt').to(device)
        tokens = encoded_input['input_ids'].to(device)
        target_ids = tokens.clone().to(device)
        loss = gpt_model(tokens, labels = target_ids).loss
        ppl_list.append(torch.exp(loss).detach().cpu().numpy())

    return ppl_list

def rank_generations(text_list, x, z, mode="abductive"):
    if "abductive" in mode:
        text_xyz = []
        text_yz = []
        for text in text_list:
            yz = text + " " + z
            xyz = x + " " + yz
            text_xyz.append(xyz)
            text_yz.append(yz)

        rank_score = get_gpt_ppl(text_xyz, "cuda")
        combined_list = list(zip(rank_score, text_xyz, text_list))
        sorted_list = sorted(combined_list, key=lambda x: x[0])
        sorted_list2 = [li[2] for li in sorted_list]
        # print(sorted_list2)
        text_candidates = [x + " " + li for li in sorted_list2]

        rank_score = get_gpt_ppl(text_candidates, "cuda")
        combined_list = list(zip(rank_score, text_candidates, sorted_list2))
        sorted_list = sorted(combined_list, key=lambda x: x[0])
        result = sorted_list[0][2]
        # print(combined_list)
    
    return result

def vocab_prune(model, y_logits, topk, x_onehot, x_past, tokenizer, extra_mask=None):
    assert x_onehot.shape[1] == 1, x_onehot.shape
    length = y_logits.shape[1]
    past = x_past
    input_embeds = torch.matmul(x_onehot.float(), model.get_input_embeddings().weight)
    mask_t_all = None
    logits_so_far = None

    for i in range(length):
        model_outputs = model(past_key_values=past, inputs_embeds=input_embeds, use_cache=True) # 送进去历史信息
        past = model_outputs.past_key_values    
        logits_t = model_outputs.logits[:, -1:, :]  # 选出来最后一个单词的logits
        assert logits_t.shape[1] == 1, logits_t.shape   
        _, indices_t = torch.topk(logits_t, topk)   # 最后一个单词的topk logits
        mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)   # 变mask
        mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)   # 第i个单词的topk-mask
        logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)  # 生成的y的logits
        if i < length - 1:
            if extra_mask is None:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
            else:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
            input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1), model.get_input_embeddings().weight)

    return logits_so_far, mask_t_all

def forw(model, y_logits, topk, x_onehot, x_past):
    # BIG_CONST = 1e10
    # assert x_onehot.shape[1] == 1, x_onehot.shape
    # length = y_logits.shape[1]
    # past = x_past
    # input_embeds = torch.matmul(x_onehot.float(), model.get_input_embeddings().weight)
    # mask_t_all = None
    # logits_so_far = None

    # for i in range(length):
    #     model_outputs = model(past_key_values=past, inputs_embeds=input_embeds) # 送进去历史信息
    #     past = model_outputs.past_key_values    
    #     logits_t = model_outputs.logits[:, -1:, :]  # 选出来最后一个单词的logits
    #     assert logits_t.shape[1] == 1, logits_t.shape   
    #     _, indices_t = torch.topk(logits_t, topk)   # 最后一个单词的topk logits
    #     mask_t = mask_[:, i:i+1, :]
    #     # logits_t[~mask_t] = -1 * BIG_CONST
    #     mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)   # 第i个单词的topk-mask

    #     logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)  # 生成的y的logits

    #     if i < length - 1:
    #         y_logits_i_topk = y_logits[:, i:i+1, :] / 0.001
    #         input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1), model.get_input_embeddings().weight)    
    
    xy_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits / 0.0001,
        x_onehot=x_onehot,
        device=x_onehot.device
    )
    # print(x_past.shape)
    # embed_inputs: [bsz, length, embed_size]
    xy_logits = model(past_key_values=x_past, inputs_embeds=xy_embeds, use_cache=True).logits
    x_length = x_onehot.shape[1]
    logits_so_far = xy_logits[:, x_length - 1:-1, :]

    return logits_so_far

def contrastive_loss(y_logits):
    # y_logits: B, L, |V|
    bsz, length, vocab_size = y_logits.shape
    norm_rep = y_logits / y_logits.norm(dim=2, keepdim=True)
    cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2))
    gold_score = torch.diagonal(cosine_scores, offset=0, dim1=1, dim2=2) # bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1)
    assert gold_score.size() == torch.Size([bsz, length, 1])
    difference_matrix = gold_score - cosine_scores
    loss_matrix = 0.5 - difference_matrix # bsz x seqlen x seqlen
    loss_matrix = torch.nn.functional.relu(loss_matrix)
    cl_loss = torch.sum(loss_matrix) / (bsz * length * vocab_size)
    return cl_loss

def find_nearest_vectors_pytorch(target_vectors, candidate_vectors, batch_size):
    # 将target_vectors扩展为(batch_size * num_targets, vector_size)
    extended_target_vectors = target_vectors.view(-1, target_vectors.size(-1))
    
    # 计算距离矩阵
    distances = torch.cdist(extended_target_vectors, candidate_vectors)
    
    # 找到最小距离对应的索引
    nearest_indices = torch.argmin(distances, dim=1)
    
    # 根据最小距离的索引获取最近的向量
    nearest_vectors = candidate_vectors[nearest_indices]
    
    # 将结果重新组织为(batch_size, num_targets, vector_size)
    nearest_vectors = nearest_vectors.view(batch_size, -1, nearest_vectors.size(-1))
    
    return nearest_vectors

def sim_score(model, y_logits, ref_vec):
    y_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits / 0.0001,
        device=y_logits.device
    )
    # print(y_embeds.grad)
    y_vec = model(inputs_embeds=y_embeds, use_cache=True, output_hidden_states=True).hidden_states[-1].mean(dim=1)
    return F.cosine_similarity(ref_vec, y_vec)

    # return -1 * bert_score(embedding_y, ref_emb)

def get_ref_embedding(model, ref, device, tokenizer):
    ref_ = tokenizer.encode(ref)[:]
    ref_ = torch.tensor(ref_)
    ref_ = ref_.to(device)
    if ref_.dim() == 1:
        ref_ = ref_.unsqueeze(0)
    
    # ref_t = torch.tensor(ref_, device=device, dtype=torch.long)
    # ref_onehot = one_hot(ref_t, dimension=tokenizer.vocab_size)

    ref_vec = model(ref_, output_hidden_states=True).hidden_states[-1].mean(dim=1).detach()
    # print(len(output.hidden_states))
    return ref_vec

def bert_score(embedding1, embedding2):
    # Normalize embeddings
    # embedding2 = embedding2.repeat(embedding1.shape[0], 1, 1)
    embedding1 = embedding1 / torch.norm(embedding1, dim=-1).unsqueeze(-1)
    embedding2 = embedding2 / torch.norm(embedding2, dim=-1).unsqueeze(-1)
    # Calculate similarity matrix
    similarity_matrix = torch.bmm(embedding1, embedding2.transpose(1, 2))
    word_precision = similarity_matrix.max(dim=2)[0]
    word_recall = similarity_matrix.max(dim=1)[0]
    P = (word_precision / word_precision.shape[-1]).sum(dim=1)
    R = (word_recall / word_recall.shape[-1]).sum(dim=1)
    F = 2 * P * R / (P + R)
    return F
    # Calculate precision, recall, and F1 score
    # precision = similarity_matrix.max(dim=2)[0]/embedding1.shape[1]
    # recall = similarity_matrix.max(dim=1)[1]/embedding2.shape[1]
    # f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)  # Adding a small epsilon to avoid division by zero

    return f1_score

def score_by_bert(A, hyps, B, model, tokenizer, device='cuda'):
    """
    Use BERT next-sentence-prediction to compute the scores of
    (A-hyps, B) and (A, hyps-B)

    Args:
        A: O1
        hyps: hypothesis
        B: O2
    """
    def _score(a, b):
        encoded = tokenizer.encode_plus(a, text_pair=b, return_tensors='pt')
        for k in encoded:
            encoded[k] = encoded[k].to(device)
        seq_relationship_logits = model(**encoded)[0]
        return (seq_relationship_logits[0, 0].tolist())

    res_A_hB = []
    res_Ah_B = []
    for hyp in hyps:
        if hyp == 'DEPRECATED':
            res_A_hB.append(-1)
            res_Ah_B.append(-1)
            continue
        hB = ' '.join([hyp, B])
        Ah = ' '.join([A, hyp])
        res_A_hB.append(_score(A, hB))
        res_Ah_B.append(_score(Ah, B))

    return res_A_hB, res_Ah_B
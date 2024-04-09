from contextlib import contextmanager
import signal
import torch as th
from torch import nn
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin, top_k_top_p_filtering
import threading
import queue
import multiprocessing
import time
from typing import Callable
from collections import defaultdict

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def run_until(seconds: int, func: Callable, *args) -> None:
    """Run a function until timeout in seconds reached."""
    with multiprocessing.Pool(processes=2) as pool:
        result = pool.apply_async(func, [*args])
        try:
            result.get(timeout=seconds)
            return result.get()
        except multiprocessing.TimeoutError:
            return None

def eval_with_timeout(formula, max_time=2):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None

def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "").strip()
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)

def calculator_sample(model, qn, tokenizer, device, max_len,
    beam_size=1, top_k=None, temperature=.7, top_p=1.0):
    '''
    Faster version of calculator sampling -- caches activations from previous tokens, no batching though (very complicated)
    Args:
        model: T5 model (encoder + decoder)
        qn: question string
        tokenizer: T5 tokenizer
        device: torch device
        max_len: maximum length of generated sequence
        beam_size: beam size for beam search
        top_k: top-k sampling
        temperature: temperature for sampling
        top_p: top-p sampling

    Returns:
        list of generated strings
    
    1. Encode the question to get the encoder outputs
    2. Initialize the decoder input with the start token
    3. generate the next token using the decoder
    4. At each step check if the last token is an equals token
    5. If it is an equals token, use the calculator to get the answer
    6. If the answer is not None, append the answer to the generated string
    7. If the answer is None, do nothing
    8. If the last token is an end token, break
    9. If the last token is not an end token, repeat from step 3


    '''
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    assert 't5' in model.config._name_or_path, "Only T5 models are supported"
    
    EQUAL_TOKENS = [3274, 2423]

    bad_words_ids = [[tokenizer.pad_token_id]]
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    enc_input_ids = tokenizer([qn], return_tensors='pt').input_ids.to(device)
    model_kwargs = {"use_cache": True}
    
    model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
        enc_input_ids, model_kwargs)

    input_ids = th.LongTensor([[tokenizer.pad_token_id]]).to(model.device)

    config = GenerationConfig(
        repetition_penalty=1.0,
        encoder_no_repeat_ngram_size=0,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beam_groups=1,
        diversity_penalty=0.0,
        remove_invalid_values=None,
        exponential_decay_length_penalty=None,
        renormalize_logits=False,
        no_repeat_ngram_size=0,
        bad_words_ids=bad_words_ids,
        max_length=max_len,
        eos_token_id=eos_token_id,
        num_beams=beam_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        )

    logits_processor = model._get_logits_processor(
            generation_config = config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=enc_input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            )
    
    logits_warper = model._get_logits_warper(config)

    cur_len = 1
    while cur_len < max_len:
        
        model_inputs = model.prepare_inputs_for_generation(
                input_ids, 
                **model_kwargs
            )
        
        outputs = model(**model_inputs, return_dict=True)
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids=input_ids, scores=next_token_logits)
        next_token_scores = logits_warper(input_ids=input_ids, scores=next_token_scores)
        
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_token = th.multinomial(probs, num_samples=1).squeeze(1)
        
        input_ids = th.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if next_token.item() in EQUAL_TOKENS:
            text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            answer = use_calculator(text)
 
            if answer is not None:
                if isinstance(answer, float):
                    ## check if integer 
                    if answer.is_integer():
                        answer = int(answer)
                
                #print("Triggered calculator, answer", answer)
                answer = str(answer) + " >>"
                answer_tokens = tokenizer.encode("=" + answer, add_special_tokens=False)[1:]
                cur_len += len(answer_tokens)
                #print("adding answer tokens:", tokenizer.decode(answer_tokens, skip_special_tokens=False))
            
                assert answer_tokens[0] != 3 and answer_tokens[-1] != 1                
                answer_tokens = th.LongTensor(answer_tokens).unsqueeze(0).to(model.device)
                input_ids = th.cat([input_ids, answer_tokens], dim=-1)
            
        elif next_token.item() == eos_token_id:
            break
        
        cur_len += 1
        #print(tokenizer.decode(input_ids[0], skip_special_tokens=False))

    return input_ids[0]


def calculator_sample_batch(model, qns, tokenizer, device, max_len,
    beam_size=1, top_k=None, temperature=.7, top_p=1.0):
    '''
    batchified version of calculator sampling. 
    Args:
        model: T5 model (encoder + decoder)
        qns: list of question strings
        tokenizer: T5 tokenizer
        device: torch device
        max_len: maximum length of generated sequence
        beam_size: beam size for beam search
        top_k: top-k sampling
        temperature: temperature for sampling
        top_p: top-p sampling

    Returns:
        list of generated ids

    1. Encode the questions to get the encoder outputs
    2. Initialize the decoder input with the start token
    3. generate the next token using the decoder
    4. At each step check if the last token is an equals token
    5. If it is an equals token, use the calculator to get the answer
    6. If the answer is not None, append the answer to the generated string
    7. If the answer is None, do nothing
    '''

    # Efficient version of calculator sampling -- uses batches, caches
    # activations from previous tokens
    assert 't5' in model.config._name_or_path, "Only T5 models are supported"

    EQUAL_TOKENS = [3274, 2423]
    bsz = len(qns)
    bad_words_ids = [[tokenizer.pad_token_id]]
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    eos_token_id_tensor = th.tensor([eos_token_id]).to(device) if eos_token_id is not None else None
    enc_input_ids = tokenizer(qns, return_tensors='pt', padding=True).input_ids.to(device)
    model_kwargs = {"use_cache": True}

    model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
        enc_input_ids, model_kwargs)

    input_ids = th.LongTensor([[tokenizer.pad_token_id]] * bsz).to(model.device)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    config = GenerationConfig(
        repetition_penalty=1.0,
        encoder_no_repeat_ngram_size=0,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beam_groups=1,
        diversity_penalty=0.0,
        remove_invalid_values=None,
        exponential_decay_length_penalty=None,
        renormalize_logits=False,
        no_repeat_ngram_size=0,
        bad_words_ids=bad_words_ids,
        max_length=max_len,
        eos_token_id=eos_token_id,
        num_beams=beam_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        )
    
    logits_processor = model._get_logits_processor(
            generation_config = config, 
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=enc_input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            )
    
    logits_warper = model._get_logits_warper(config)
    
    cur_len = 1
    idx_to_next_token = defaultdict(list) # maps idx to list of next tokens to be used in the next step. 
    
    while cur_len < max_len:
        
        model_inputs = model.prepare_inputs_for_generation(
                input_ids, 
                **model_kwargs
            )
        
        print("input_ids", input_ids.size())
        if input_ids.shape[1] == 162:
            import ipdb; ipdb.set_trace()
        outputs = model(**model_inputs, return_dict=True)
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids=input_ids, scores=next_token_logits)
        next_token_scores = logits_warper(input_ids=input_ids, scores=next_token_scores)
        
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_token = th.multinomial(probs, num_samples=1).squeeze(1)

        ## replace next token with the next token in the list if it exists
        #for i, token in enumerate(next_token):
        #    if idx_to_next_token[i]:
        #        next_token[i] = idx_to_next_token[i].pop(0)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_token = next_token * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = th.cat([input_ids, next_token.unsqueeze(1)], dim=-1)
        
        # check if any of the generated tokens are equal tokens
        '''
        for i, token in enumerate(next_token):
            if token.item() in EQUAL_TOKENS:
                text = tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=True)
                answer = use_calculator(text)
 
                if answer is not None:
                    if isinstance(answer, float):
                        ## check if integer 
                        if answer.is_integer():
                            answer = int(answer)
                
                    #print("Triggered calculator, answer", answer)
                    answer = str(answer) + " >>"
                    answer_tokens = tokenizer.encode("=" + answer, add_special_tokens=False)[1:]
                    
                    ## add answer tokens to the idx list 
                    idx_to_next_token[i].extend(answer_tokens)
        '''
        
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_token.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

        print("finished sequences", 1 - unfinished_sequences)
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0:
            break

        cur_len += 1

    return input_ids


        

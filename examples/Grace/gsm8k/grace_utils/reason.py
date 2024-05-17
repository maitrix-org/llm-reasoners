from tqdm import tqdm
import torch
import re, time
from data_utils.utils import use_calculator
from constants import *
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import top_k_top_p_filtering
from torch.nn.utils.rnn import pad_sequence

DELIMITER_TO_ID_T5 = {
    '|': 1820,
    ';': 117,
    '. ': 5
}

DELIMITER_TO_ID_LLAMA = {
    '|': 891,
    '. ': 29889,
    ';': 29936
}

def generate_guided_reasoning(model, 
            model_tokenizer, 
            discriminator, 
            disc_tokenizer,
            model_input_text, 
            disc_input_text, 
            n_candidate_steps=10, 
            beta=0.5,
            generation_type='token',
            args=None
            ):
    """ At each timestep t. we generate J candidate steps, they represent the multiple choices and then we score each choice, here candidate steps denotes the number of steps generated from  nucleus sampling."""
    if not isinstance(model_input_text, list):
        model_input_text = [model_input_text]
    
    if not isinstance(disc_input_text, list):
        disc_input_text = [disc_input_text]

    is_enc_dec = hasattr(model, 'get_encoder')

    with torch.no_grad():
        # assumes initially all same length.
        model_input_ids = [model_tokenizer.encode(it, return_tensors='pt').to(model.device) for it in model_input_text] # batch x seq
        model_input_ids = torch.cat(model_input_ids, dim=0)

        input_ids = torch.LongTensor([[model_tokenizer.pad_token_id]] * len(model_input_text)).to(model.device) # batch x seq
        cur_len = 1
        temperature = args.temperature
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = [[model_tokenizer.pad_token_id]]
        pad_token_id = model_tokenizer.pad_token_id
        eos_token_id = model_tokenizer.eos_token_id
        attention_mask = model_input_ids.new_ones(model_input_ids.shape)
        use_cache = True
        
        if is_enc_dec:
            model_specific_kwargs = {'encoder_outputs': model.get_encoder()(model_input_ids, attention_mask=attention_mask)}
        else:
            model_specific_kwargs = {}
        
        #### prepare discriminator input (if needed) ####
        if generation_type in ['token', 'step']:
            disc_input_ids = [disc_tokenizer.encode(it, return_tensors='pt').to(discriminator.device) for it in disc_input_text]
            disc_input_ids = torch.cat(disc_input_ids, dim=0)

        #### step delimiter depending on model type ####
        step_delimiter = getattr(args, 'step_delimiter', '|')
        if 'T5' in model_tokenizer.__class__.__name__:
            step_delimiter_id = DELIMITER_TO_ID_T5[step_delimiter]
        elif 'Llama' in model_tokenizer.__class__.__name__:
            step_delimiter_id = DELIMITER_TO_ID_LLAMA[step_delimiter]
        else:
            raise NotImplementedError(f"step delimiter ID not set for {model_tokenizer.__class__.__name__}")

        assert step_delimiter not in model_tokenizer.all_special_tokens, "step delimiter cannot be a special token!!!"

        output = _generate_step_with_score_disc(model=model,
                    discriminator=discriminator,
                    beta=beta,
                    n_candidate_steps=n_candidate_steps,
                    decoder_input_ids=input_ids,
                    model_input_ids=model_input_ids,
                    cur_len=cur_len,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    step_delimiter_id=step_delimiter_id,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    model_kwargs=model_specific_kwargs,
                    question = disc_input_text[0],
                    model_tokenizer=model_tokenizer,
                    disc_tokenizer=disc_tokenizer,
                    args=args)
        
        return [model_tokenizer.decode(s, skip_special_tokens=True) for s in output]

def _generate_step_with_score_disc(
        model,
        discriminator,
        beta,
        n_candidate_steps,
        model_input_ids,
        decoder_input_ids,
        cur_len,
        temperature,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        step_delimiter_id,
        attention_mask,
        use_cache,
        model_kwargs,
        question,
        model_tokenizer=None,
        disc_tokenizer=None,
        args=None
    ):
        """generated solutions using step-wise discriminator that was trained as a scoring function
         The guided stepwise decoding process using the trained discriminator.
        Given the question and the prefix, we sample a pool of candidate next steps and use the discriminator to score steps. 
        The top-scored step is then selected and added to the prefix. This process repeats until a final answer is generated."""
        # length of generated sentences / unfinished sentences
        ## encode disc_input_ids using the discrimiantor encoder to avoid re-computing 


        assert disc_tokenizer.cls_token is not None, "Score-based discriminator Tokenizer must have a [CLS] token!"
        assert disc_tokenizer.sep_token is not None, "Score-based discriminator tokenizer must have a [SEP] token!"

        prefix_ids = [] # empty prefix to begin with
        
        CALC_TOKENS = [disc_tokenizer.encode(s, add_special_tokens=False)[0] for s in ['<<', '>>']]
        assert disc_tokenizer.unk_token_id not in CALC_TOKENS, "Calculator tokens are not identified by the discriminator tokenizer!"

        ### 1. sample candidate next steps from the model with beam search
        ### 2. for each candidate, calculate the discriminator score
        ### 3. sample pick the most likely candidate according to the discriminator score
        ### 4. repeat until the end of the solution

        max_steps = args.max_steps 
        sample_method = args.step_sampling_method

        #question_ids = torch.tensor(disc_tokenizer.encode(question, add_special_tokens=False))
        question_ids = disc_tokenizer.encode(question, add_special_tokens=False)
        is_enc_dec = hasattr(model, "get_encoder")

        ## cache encoder_outputs
        if is_enc_dec:
            _encoder_outputs = model.get_encoder()(model_input_ids.repeat_interleave(1, dim=0), return_dict=True)
            _last_hidden_state = _encoder_outputs["last_hidden_state"].clone()
            model_kwargs = {"encoder_outputs": _encoder_outputs}
            cur_prefix = decoder_input_ids
        else: # decoder-only models
            model_kwargs = {}
            cur_prefix = model_input_ids
        
        # #### id
        if 'T5' in model_tokenizer.__class__.__name__:
            AND_IDENTIFIER_ID = 30345 
        elif 'Llama' in model_tokenizer.__class__.__name__:
            AND_IDENTIFIER_ID = 3191 
        else:
            raise NotImplementedError("tokenizer {} not supported!".format(model_tokenizer.__class__.__name__))
        
        original_input_length = cur_prefix.shape[1]
        
        for _ in tqdm(range(max_steps), disable=True): 
            print(f"{'-'*30} \n New loop")
            decoder_input_seq_len = cur_prefix.shape[1]
            
            all_new_sequences = [] 
            all_seq_scores = []
            sampling_bsz = getattr(args, "sampling_batch_size", n_candidate_steps)

            for i in range(0, n_candidate_steps, sampling_bsz):
                n_to_sample = min(sampling_bsz, n_candidate_steps - i)

                outputs = model.generate(
                    decoder_input_ids=cur_prefix if is_enc_dec else None,
                    input_ids=model_input_ids if is_enc_dec else cur_prefix,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_step_length,
                    do_sample=False if sample_method == "beam" else True,
                    temperature=temperature,
                    top_p=args.top_p if sample_method == "top_p" else 1.0,
                    top_k=args.top_k if sample_method == "top_k" else None,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=pad_token_id,
                    eos_token_id=[step_delimiter_id],
                    num_beams=n_to_sample if sample_method == "beam" else 1,
                    num_return_sequences=n_to_sample,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=use_cache,
                    sample_calc=args.sample_calc,
                    tokenizer=model_tokenizer,
                    **model_kwargs,
                )

                
                
                if is_enc_dec:
                    model_kwargs["encoder_outputs"]["last_hidden_state"] = _last_hidden_state.clone() ## reset the encoder outputs to have a batch size of 1
                
                sequences = outputs.sequences # batch x seq
                new_sequences = sequences[:, decoder_input_seq_len:] # batch x seq

               
                if sample_method == "beam":
                    ### get only new tokens
                    seq_scores = outputs.sequences_scores
                    ## softmax the scores
                    seq_scores = torch.softmax(seq_scores, dim=-1) # batch x seq
                
                elif sample_method in ["top_p", "top_k", "random"]:
                    transition_scores = model.compute_transition_scores(
                        outputs.sequences,
                        outputs.scores,
                        normalize_logits=True)# batch x seq
                    
                    assert transition_scores.shape[1] == new_sequences.shape[1], "Transition scores and sequences mismatch!"

                    ## normalize by length: exp()
                    probs = torch.exp(transition_scores) # batch x seq
                    logprobs = torch.log(probs) # batch x seq
                    ## divide by length of each sequence
                    seq_lens = torch.sum(new_sequences != pad_token_id, dim=-1).unsqueeze(-1) # batch x 1
                    logprobs = logprobs / seq_lens # batch x seq
                    ### set -inf to 0
                    logprobs[logprobs == float('-inf')] = 0.0
                    seq_scores = torch.exp(torch.sum(logprobs, dim=-1))
                    #seq_scores = torch.ones(sequences.shape[0]).to(sequences.device) # batch x seq

                all_new_sequences.extend(new_sequences)
                all_seq_scores.append(seq_scores)
                print(f"new_sequences shape: {new_sequences.shape}")
                print(f" all_new_sequences shape: {len(all_new_sequences)} and individual entry is : {all_new_sequences[0].shape}")
                
                sequences_list = new_sequences.tolist()

                # Decode each sequence
                decoded_sequences = [model_tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences_list]

                # Print each decoded sequence
                for i, seq in enumerate(decoded_sequences):
                    print(f"Sequence {i+1}: {seq}")

            new_sequences = pad_sequence(all_new_sequences, batch_first=True, padding_value=pad_token_id)
            seq_scores = torch.cat(all_seq_scores, dim=0)

            assert new_sequences.shape[0] == seq_scores.shape[0], "new_sequences and seq_scores mismatch!"
            assert new_sequences.shape[0] == n_candidate_steps, "new_sequences and n_candidate_steps mismatch!"

            ### check if all sequences contain a final answer
            is_all_answers = torch.all(torch.sum(new_sequences == AND_IDENTIFIER_ID, dim=1))

            if beta > 0.0 and not is_all_answers:
                # checks if there is a beta score to factor in the discriminator score in addition to the generator scores.
                # why using not is_all_answers?
                disc_input_ids = []
                prefix_ids_disc = prefix_ids

                if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                    prefix_ids_disc = model_tokenizer.decode(prefix_ids_disc, skip_special_tokens=True)
                    prefix_ids_disc = disc_tokenizer.encode(prefix_ids_disc, add_special_tokens=False)
                
                for seq in new_sequences:
                    seq = seq.tolist()
                    ### if the two tokenizers are different, we need to convert the sequence to the discriminator tokenizer
                    if model_tokenizer.__class__.__name__ != disc_tokenizer.__class__.__name__:
                        seq = model_tokenizer.decode(seq, skip_special_tokens=True)
                        seq = disc_tokenizer.encode(seq, add_special_tokens=False)
                    
                    disc_input_ids.append([disc_tokenizer.cls_token_id] + question_ids + prefix_ids_disc + [disc_tokenizer.sep_token_id] + seq)
                
                ## pad the sequences
                disc_input_ids = pad_sequence([torch.tensor(t) for t in disc_input_ids], batch_first=True, padding_value=disc_tokenizer.pad_token_id).to(discriminator.device) # batch x seq
                disc_attention_mask = disc_input_ids != disc_tokenizer.pad_token_id # batch x seq
                ## feed to discriminator to obtain scores
                disc_scores = discriminator.forward_scores(input_ids=disc_input_ids, attention_mask=disc_attention_mask).view(-1)
                
                if args.normalize_disc_scores:
                    disc_scores = torch.softmax(disc_scores, dim=-1).to(seq_scores.device) # batch
                
                assert disc_scores.shape == seq_scores.shape, "Discriminator scores shape mismatch!"
                ## calculate the final score for each sequence by combining 
                final_scores = (1 - beta) * seq_scores + beta * disc_scores # batch
            
            else:
                final_scores = seq_scores

             # the following code is to choose one sequence from the generated sequences.
            if args.step_selection_method == "greedy":
                ## sample the next step with the highest score
                next_step_idx = torch.argmax(final_scores, dim=-1) # batch x 1

            elif args.step_selection_method == "sample":
                ## sample the next step with probability proportional to the score
                next_step_idx = torch.multinomial(final_scores, num_samples=1)
            else:
                raise ValueError("Invalid step selection method!")
            
            next_step = new_sequences[next_step_idx] # (next step is of size [x])
            ## remove padding from the next step
            next_step = next_step[next_step != pad_token_id] # (next step is of size [x])
            ## update the decoder_input_ids
            cur_prefix = torch.cat([cur_prefix, next_step.unsqueeze(0)], dim=-1) # size [1, x]
            ## update attention mask if necessary
            if not is_enc_dec:
                attention_mask = cur_prefix != pad_token_id # batch x seq
            
            ## check of eos token is in the next step
            eos_in_sents = next_step == eos_token_id # batch x 1
            prefix_ids += next_step.tolist() # prefix ids is of len 60

            print("Generated step: ", model_tokenizer.decode(next_step.tolist(), skip_special_tokens=True))
            
            if eos_in_sents.sum() > 0 or is_all_answers or AND_IDENTIFIER_ID in next_step.tolist():
                break

            cur_len += 1

        if not is_enc_dec: # remove the input prefix from the generated sequence
            cur_prefix = cur_prefix[:, original_input_length:]

        
        print(f" curr prefix : {cur_prefix.shape}")
        return cur_prefix

import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import namedtuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from constants import *
from grace_utils.t5_discriminator import T5Discriminator, T5EnergyDiscriminator
# from verifier_utils.t5_verifier import T5Verifier
from grace_utils.reason import generate_guided_reasoning
import json
from data_utils.utils import prepare_icl_input, create_demos, is_correct, extract_answer, is_correct_program, extract_answer_program, extract_answer_llc
import wandb
from collections import Counter
from data_utils.utils import evaluate
from torch.nn.utils.rnn import pad_sequence
from grace_utils.args import TASKS

def main(args):
    if args.model_tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer_path)
    

    ## set padding_side to left for llama tokenizer
    if 'llama' in args.model_name_or_path:
        tokenizer.padding_side = 'left'
    
    print("Loading model from {}".format(args.model_name_or_path))
    if 't5' in args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, 
                                                       torch_dtype = torch.bfloat16 if args.bf16 else torch.float32).to(args.device1)
    elif 'llama' in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                       torch_dtype = torch.bfloat16 if args.bf16 else torch.float32,
                                                       load_in_8bit=args.eightbit,device_map="auto" if args.eightbit else None)
                                                       
        tokenizer.pad_token = tokenizer.eos_token
    
    if not args.eightbit:
        model = model.to(args.device1)
    
    model.eval()

    if args.disc_path is not None and not args.generator_only:
        ## load discriminator tokenizer 
        print("loading discriminator tokenizer from {}".format(args.disc_path))
        disc_tokenizer = T5Tokenizer.from_pretrained(args.disc_path)
        
        print("Loading discriminator from {}".format(args.disc_path))
        if os.path.isdir(args.disc_path):
            for _, _, files in os.walk(args.disc_path):
                for fname in files:
                    if 'best' in fname or 'pytorch_model.bin' in fname:
                        args.disc_path = os.path.join(args.disc_path, fname)
                        break
        
        ckpt = torch.load(args.disc_path)
        ## loading discriminator weights
        if args.generation_type == 'step':
            discriminator = T5Discriminator(model_name_or_path=ckpt['args'].model, 
            device=args.device2, args=ckpt['args'])
            discriminator.t5.resize_token_embeddings(len(disc_tokenizer))
            discriminator.load_state_dict(ckpt['state_dict'])
        elif args.generation_type in ['step-score', 'step-qrs']:
            disc_backbone = 'google/flan-t5-large'
            discriminator = T5EnergyDiscriminator(model_name_or_path=disc_backbone, 
            device=args.device2, args=args)
            discriminator.model.resize_token_embeddings(len(disc_tokenizer))
            discriminator.load_state_dict(ckpt)

        elif args.generation_type == 'token':
            assert len(tokenizer) == len(disc_tokenizer) - 2, "model and discriminator tokenizers are not the same. They have to be the same for token-level generation"

        #if len(tokenizer) != len(disc_tokenizer) - 2: # two for [SEP] and [CLS]
        #    print("WARNING: model and discriminator tokenizers are not the same. Will use a hack to make it work")
        
        discriminator.eval()

    # if args.use_verifier:
    #     assert args.verifier_path is not None
    #     ## load verifier tokenizer 
    #     print("loading verifier tokenizer from {}".format(args.verifier_path))
    #     verifier_tokenizer = T5Tokenizer.from_pretrained(args.verifier_path)
        
    #     print("Loading verifier from {}".format(args.verifier_path))
    #     if os.path.isdir(args.verifier_path):
    #         for _, _, files in os.walk(args.verifier_path):
    #             for fname in files:
    #                 if 'best' in fname or 'pytorch_model.bin' in fname:
    #                     args.verifier_path = os.path.join(args.verifier_path, fname)
    #                     break
        
    #     ckpt = torch.load(args.verifier_path)
    #     verifier_backbone = 'google/flan-t5-base' 
    #     if 'large' in args.verifier_path:
    #         verifier_backbone = 'google/flan-t5-large'
    #     if 'small' in args.verifier_path:
    #         verifier_backbone = 'google/flan-t5-small'
        
    #     verifier = T5Verifier(model_name_or_path=verifier_backbone, 
    #     device=args.device2)
    #     verifier.model.resize_token_embeddings(len(verifier_tokenizer))
    #     verifier.load_state_dict(ckpt)
    #     verifier.eval()

    demos = None
    eval_examples = []
    with open(args.in_file, 'r') as rf:
        for line in rf:
            d = json.loads(line)
            eval_examples.append(d)
    
    n_examples = args.n_examples if args.n_examples is not None else len(eval_examples)
    end_idx = min(args.start_idx + n_examples, len(eval_examples))
    eval_examples = eval_examples[args.start_idx:end_idx]
    
    if args.icl:
        ## assert demos are not in the inputs
        demos = []
        ## load demos 
        print("Using In-context learning with {} demos".format(args.n_demos))
        ## in file location 
        data_path = os.path.join(*args.in_file.split('/')[:-1])
        demos_path = os.path.join(data_path, args.demos_file_name)

        with open(demos_path, "r") as f:
            for line in f:
                demo = json.loads(line)
                demos.append(demo)
                assert demo['question'] not in [a['question'] for a in eval_examples], "The demo {} is in the eval examples!".format(demo['question'])
        ## process demos
        demos = create_demos(demos, step_delimiter=args.step_delimiter, add_delimiter_after_demo=True)[:args.n_demos]

    #inputs = [inputs[1]]
    #gt_outputs = [gt_outputs[1]]
    print("Task: {}".format(args.task))

    text_table = wandb.Table(columns=["question", "gold answer", "solution"])
    
    if args.generator_only:
        print("Evaluating generator only...")
        eval_d = evaluate(model, tokenizer, eval_examples, demos=demos, instruction=args.instruction, 
        task=args.task, args=args)
        print("solve rate w/o discriminator: {:.2f}".format(eval_d['eval_acc'] * 100))

        wandb.log({"solve_rate": eval_d['eval_acc']})
        for ex, gen in zip(eval_examples, eval_d['generated_solutions']):
            text_table.add_data(ex['question'], ex['answer'], gen)
        
        wandb.log({"outputs": text_table})
        return

    if args.task in ["mathqa"]:
        assert args.step_delimiter.strip() == ';', "Step delimiter for {} should be ;"
    elif args.task in ["gsm8k", "svamp", "multiarith", "coin_flip"] and not args.icl:
        ## not few-shot 
        assert args.step_delimiter.strip() == '|', "Step delimiter for {} should be |".format(args.task)
    
    extract_answer_fn = None 
    if args.task in ['arithmetic', 'gsm8k', 'svamp', 'multiarith', 'coin_flip']:
        extract_answer_fn = extract_answer
    elif args.task in ['asdiv', 'mathqa']:
        extract_answer_fn = extract_answer_program
    elif args.task in ['last_letter_concatenation']:
        extract_answer_fn = extract_answer_llc
    else:
        raise NotImplementedError("Task {} not implemented!".format(args.task))
    

    print("Running guided decoding...")

    solve_rate = 0

    if args.use_verifier:
        all_solutions = []

    count=0
    for inp in tqdm(eval_examples):
        count+=1
        # if count>1:
        #     break
        qn, gt_ans = inp['question'], (inp['answer'] if 'answer' in inp else inp['gt_sol'])

        qn_with_input = prepare_icl_input(qn, demos=demos, instruction=args.instruction)
        disc_input = qn
        if not disc_input.startswith(Q_DELIM):
            disc_input = Q_DELIM + " " + disc_input
        
        n_samples = args.n_verifier_samples if args.use_verifier else args.n_self_consistency
        gen_sols = []

        print(f"qn_with_input: {qn_with_input}")
        print(f"disc_input: {disc_input}")
        print(f"num_samples: {n_samples}")
        for _ in range(n_samples):
            results = generate_guided_reasoning(
                            model=model, 
                            model_tokenizer=tokenizer,
                            discriminator=discriminator,
                            disc_tokenizer = disc_tokenizer,
                            model_input_text=qn_with_input,
                            disc_input_text=disc_input, 
                            n_candidate_steps=args.n_candidate_steps,
                            beta=args.beta,
                            generation_type=args.generation_type,
                            args=args,
                            )
    
            sol = results[0]
            gen_sols.append(sol)
        print(f"solution: {sol}")
        if count >5:
            raise Exception
        
        if not args.use_verifier: ## self-consistency
            ## extract answers 
            answers = [extract_answer_fn(x) for x in gen_sols]
            answers = [a for a in answers if (a is not None and a != '[invalid]')]
            
            if len(answers) == 0: # all solutions are invalid -- do not have a final answer
                gen_sol = gen_sols[0] # pick any solution
            else:
                ## pick the majority answer
                voted_answer = Counter(answers).most_common(1)[0][0]
                ## pick the first solution that has the voted answer
                gen_sol = None
                for i in range(len(gen_sols)):
                    if extract_answer_fn(gen_sols[i]) == voted_answer:
                        gen_sol = gen_sols[i]
                        break
            
            assert gen_sol is not None
            if is_correct(gen_sol, gt_ans, task=args.task):
                solve_rate += 1
            text_table.add_data(qn, gt_ans, gen_sol)
        
        else:
            all_solutions.append({"question": qn_with_input, 
                                    "solutions": gen_sols,
                                    "gt_ans": gt_ans})
                                
    if args.use_verifier:
        ## rank all solutions using verifier 
        assert len(all_solutions) == len(eval_examples)
        verifier_inputs = []

        for i in tqdm(range(len(all_solutions))):
            question = all_solutions[i]['question']
            gen_sols = all_solutions[i]['solutions']
            question_ids = verifier_tokenizer.encode(question, add_special_tokens=False)

            for sol in gen_sols:
                sol_tokens = verifier_tokenizer.encode(sol, add_special_tokens=False)
                tokens = [verifier_tokenizer.cls_token_id] + question_ids + [verifier_tokenizer.sep_token_id] + sol_tokens
                verifier_inputs.append(tokens)

        assert len(verifier_inputs) == len(eval_examples) * args.n_verifier_samples
        ## pad verifier input ids

        verifier_input_ids = pad_sequence([torch.tensor(x) for x in verifier_inputs], batch_first=True, padding_value=verifier_tokenizer.pad_token_id).to(args.device2)
        verifier_attention_mask = (verifier_input_ids != verifier_tokenizer.pad_token_id).long()

        correct_scores = []

        ## feed to verifier and obtain scores
        with torch.no_grad():
            ## feed to verifier and obtain scores using args.batch_size
            bsz = args.verifier_batch_size
            for i in range(0, verifier_input_ids.shape[0], bsz):
                verifier_input_ids_batch = verifier_input_ids[i:i+bsz]
                verifier_attention_mask_batch = verifier_attention_mask[i:i+bsz]
                scores = verifier(input_ids=verifier_input_ids_batch, 
                                attention_mask=verifier_attention_mask_batch)
                scores = scores.cpu().numpy()
                scores = scores.reshape(-1, 2)
                correct_scores.extend(scores[:, 1])

        assert len(correct_scores) == len(verifier_inputs)
        correct_scores = np.array(correct_scores).reshape(-1, args.n_verifier_samples)

        ## pick the best solution for each example
        correct_idx = np.argmax(correct_scores, axis=1)

        for i in range(len(eval_examples)):
            qn, gt_ans = eval_examples[i]['question'], eval_examples[i]['answer'] if 'answer' in eval_examples[i] else eval_examples[i]['gt_sol']
            gen_sols = all_solutions[i]['solutions']
            best_sol_idx = correct_idx[i]
            gen_sol = gen_sols[best_sol_idx]

            if is_correct(gen_sol, gt_ans, task=args.task):
                solve_rate += 1
            
            text_table.add_data(qn, gt_ans, gen_sol)


    # wandb.log({"solve_rate": solve_rate / (len(eval_examples) + 1e-10)})
    # wandb.log({"outputs": text_table})
    for row in text_table.data:
        print()
        formatted_row = " | ".join(str(item) for item in row)
        print(formatted_row)
    print("Solve rate = {:.2f}".format(solve_rate*100 / len(eval_examples)))
    



if __name__=='__main__':
    parser = ArgumentParser()
    # DATA
    parser.add_argument('--disc_path', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default='google/flan-t5-large')
    parser.add_argument('--model_tokenizer_path', type=str, default=None)
    parser.add_argument('--in_file', type=str, default=None, required=True, help='file containing text to run pred on')
    parser.add_argument('--task', type=str, default='gsm8k', choices=TASKS)
    parser.add_argument('--n_candidate_steps', type=int, default=20, help='number of candidate steps to sample and score')
    parser.add_argument('--beta', type=float, default=0.85, help='weight of the discriminator score')
    parser.add_argument('--max_length', type=int, default=256, help='max length')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device1', type=str, default='cuda:0')
    parser.add_argument('--device2', type=str, default='cuda:1')

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    ## ICL 
    parser.add_argument('--icl', action='store_true', default=False, help='use icl')
    parser.add_argument('--n_demos', type=int, default=2, help='number of demonstrations')
    parser.add_argument('--demos_file_name', type=str, default='demos.jsonl', help='file containing demonstrations')
    parser.add_argument('--instruction', type=str, default=None, help='instruction to prepend to input')
    ## other
    parser.add_argument('--n_examples', type=int, default=None, help='number of examples to run')
    ### discriminator/sampling stuff
    parser.add_argument('--disc_icl', action='store_true', default=False, help='use icl with discriminator')
    parser.add_argument('--bf16', action='store_true', default=False, help='use bf16')
    parser.add_argument('--eightbit', default=False, type= lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--generation_type', type=str, default='token', choices=['token', 'step', 'step-score', 'step-qrs'])
    parser.add_argument('--disc_step_score_aggregation', type=str, default='mean', choices=['mean', 'max', 'formula', 'delimiter'])
    parser.add_argument('--step_sampling_method', type=str, default='beam', choices=['beam', 'top_p', 'top_k', 'random'])
    parser.add_argument('--max_steps', type=int, default=8, help='max number of steps to run')
    parser.add_argument('--max_step_length', type=int, default=100, help='max length of each step')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p (stepwise) sampling')
    parser.add_argument('--top_k', type=int, default=50, help='top k (setepwise) sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling')
    parser.add_argument('--output_results_dir', type=str, default=None, help='output results dir')
    parser.add_argument('--sample_calc', default=True, type= lambda x: (str(x).lower() in ['true','1', 'yes']),
                         help='whether to use calculator for sampling (mainly for GSM8K')
    parser.add_argument('--step_selection_method', type=str, default='greedy', choices=['greedy', 'sample'], help="how to select next step after ranking")
    parser.add_argument('--step_delimiter', type=str, default='|', choices=['|', '. ', ';'], help='delimiter to use for stepwise sampling')
    parser.add_argument('--n_self_consistency', type=int, default=1, help='number of samples to use for majority voting')
    parser.add_argument('--normalize_disc_scores', default=True, type= lambda x: (str(x).lower() in ['true','1', 'yes']), help='whether to normalize discriminator scores over candidate steps')
    ## further sampling params
    parser.add_argument('--goal', type=str, default='eval', choices=['eval', 'sample'], help="whether to solve or sample trajectories for further training")
    parser.add_argument('--n_samples_per_example', type=int, default=10, help='number of samples to generate per question')
    parser.add_argument('--start_idx', type=int, default=0, help='starting question idx for sampling')
    ## save dir
    parser.add_argument('--out_dir', type=str, default=None, help='save dir')

    ### generator only stuff
    parser.add_argument('--generator_sampling_method', type=str, default='greedy', choices=['beam', 'greedy'])
    parser.add_argument('--generator_beam_size', type=int, default=3, help='beam size')
    parser.add_argument('--generator_batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--generator_only', default=False, 
                        type= lambda x: (str(x).lower() in ['true','1', 'yes']),
                         help='generator only without CAD')
    ### verifier stuff 
    parser.add_argument('--use_verifier', default=False, type= lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--verifier_path', type=str, default=None, help='verifier model path')
    parser.add_argument('--n_verifier_samples', type=int, default=5, help='number of samples to use for verification')
    parser.add_argument('--verifier_batch_size', type=int, default=32, help='batch size for verification')
                        
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## print args
    print('args', json.dumps(vars(args), indent=4, sort_keys=True))

    main(args)

import json
import os
import re
import sys 
import numpy as np
from contextlib import contextmanager
import signal
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

sys.path.append('../')

from constants import A_DELIM, Q_DELIM, DEMO_SEP, ANS_IDENTIFIER

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_examples(split, dir=None, add_delim=True):
    path = os.path.join(dir, f"{split}.jsonl")
    examples = read_jsonl(path)
    print(f"{len(examples)} {split} examples")
    return examples

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
LLC_ANS_RE = re.compile(r"#### ([a-zA-Z]+)")

INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def extract_answer_llc(completion):
    match = LLC_ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion, gt_answer, task='gsm8k'):
    if task in ['gsm8k', 'svamp', 'multiarith']:
        gt_answer = extract_answer(gt_answer).rstrip(".") # remove trailing period (if any)
        assert gt_answer != INVALID_ANS
        try:
            model_ans = float(extract_answer(model_completion).rstrip("."))
            gt_ans = float(gt_answer)
            return abs(model_ans - gt_ans) < 1e-4
        
        except:
            return False
    
    elif task in ['last_letter_concatenation', 'coin_flip', 'tso']:
        gt_ans = extract_answer_llc(gt_answer)
        assert gt_ans != INVALID_ANS
        try:
            model_ans = extract_answer_llc(model_completion)
            return gt_ans == model_ans
        except:
            return False
    
    elif task in ['mathqa', 'asdiv']:
        if ANS_IDENTIFIER in gt_answer:
            gt_answer = float(extract_answer(gt_answer))
            ## remove final answer to evaluate program separately
            program = model_completion.split('####')[0].lstrip(A_DELIM).strip()
            program = program.replace('answer', 'ans')
            program = program.replace("print(ans);", "") # remove print statement
            print('program: ', program)
            try:
                loc = {}
                with timeout(3, program):
                    exec(program, globals(), loc)
                    if np.isclose(float(loc['ans']), gt_answer):
                        print(loc['ans'])
                        return True
                    else:
                        return False
            except Exception as e:
                signal.alarm(0)
                #print(f"Warning: Failed to eval {program}, exception: {e}")
                return False
    else:
        raise NotImplementedError(f"is_correct not implemented for task {task}")

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None

def is_correct_program(model_completion, gt_answer, task='mathqa'):
    assert task in ['mathqa', 'asdiv']
    if ANS_IDENTIFIER in gt_answer:
        gt_answer = float(extract_answer(gt_answer))
    ## remove final answer to evaluate program separately
    program = model_completion.split('####')[0].lstrip(A_DELIM).strip()
    program = program.replace("print(ans);", "") # remove print statement
    try:
        loc = {}
        with timeout(3, program):
            exec(program, globals(), loc)
            if np.isclose(float(loc['ans']), gt_answer):
                return True
            else:
                return False
    except Exception as e:
        signal.alarm(0)
        #print(f"Warning: Failed to eval {program}, exception: {e}")
        return False

def extract_answer_program(program):
    ## remove final answer to evaluate program separately
    program = program.split('####')[0].lstrip(A_DELIM).strip()
    program = program.replace("print(ans);", "") # remove print statement
    program = program.replace("print(answer);", "") # remove print statement
    try:
        loc = {}
        with timeout(3, program):
            exec(program, globals(), loc)
            try:
                f = float(loc['ans'])
                return f
            except:
                return '[invalid]'
    except Exception as e:
        signal.alarm(0)
        #print(f"Warning: Failed to eval {program}, exception: {e}")
        return '[invalid]'

    
def create_demos(examples, step_delimiter=None, add_delimiter_after_demo=False):
    demos = []
    step_delimiter = step_delimiter.strip()
    if step_delimiter not in ['.', ';'] and step_delimiter is not None:
        step_delimiter = ' ' + step_delimiter.strip() + ' ' ## add spaces around
    
    for ex in examples:
        ex_q, ex_a = ex['question'], ex['answer']
        if step_delimiter is not None:
            if step_delimiter.strip() == ';':
                ex_a = ex_a.replace('\n', ';')
            elif step_delimiter not in ['.', ';']:
                ex_a_steps = sent_tokenize(ex_a)
                ex_a_steps = [s.rstrip('.') for s in ex_a_steps]
                ex_a = step_delimiter.join(ex_a_steps)
        demos.append(" ".join([Q_DELIM, ex_q, A_DELIM, ex_a]))
        if add_delimiter_after_demo:
            demos[-1] += " " + step_delimiter.strip()
    return demos


def prepare_icl_input(problem, demos, instruction=None):
    ## remove whitespace from demos  # TODO handle step delimiter
    prompt = ""
    if demos is not None and len(demos) > 0:
        prompt = DEMO_SEP.join(demos) + DEMO_SEP
    
    prompt +=  " ".join([Q_DELIM, problem])
    prompt = ' '.join(prompt.split()) # remove extra white spaces
    if instruction is not None and instruction.strip() != "":
        prompt = instruction + " " + prompt
    return prompt


def process_output_sol(sol):
    sol = sol[:sol.find]

def strip_computations(sol):
    ## removes << xxx >> from the solution
    return re.sub(r"<<.*?>>", "", sol)

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


def eval_with_timeout(formula, max_time=3):
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
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0].strip()
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


def evaluate(
    model,
    tokenizer,
    eval_examples,
    demos=[],
    instruction='',
    task='gsm8k',
    args=None
    ):
        ## greedy/beam evaluation -- baseline 

        bsz = args.generator_batch_size if hasattr(args, 'generator_batch_size') else args.batch_size
        sampling_method = args.generator_sampling_method if hasattr(args, 'generator_sampling_method') else args.sampling_method
        beam_size = args.generator_beam_size if hasattr(args, 'generator_beam_size') else args.beam_size
        sample_calc = args.sample_calc

        all_generated = []
        
        acc = 0
        for i in tqdm(range(0, len(eval_examples), bsz)):
            batch_text = [a['question'] for a in eval_examples[i:i+bsz]]
            batch_text = [prepare_icl_input(t, demos=demos, instruction=instruction) for t in batch_text]
            batch = tokenizer(batch_text, padding=True, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            generated = model.generate(**batch, 
                                        max_new_tokens=args.max_length, 
                                        num_beams=1 if sampling_method=='greedy' else beam_size,
                                        do_sample=False, 
                                        temperature=1.0, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        pad_token_id=tokenizer.pad_token_id, 
                                        tokenizer=tokenizer, sample_calc=sample_calc)
            ## if decoder-only model, remove the input text
            if not hasattr(model, 'get_encoder'):
                generated = generated[:, batch["input_ids"].shape[1]:]
            
            generated = tokenizer.batch_decode(generated, skip_special_tokens=True)
            gt_ans = [ex["answer"] for ex in eval_examples[i:i+bsz]]
            all_generated.extend(generated)
            #print(generated[:5])

            if task in ['asdiv', 'mathqa']:
                # execute programs in generated text and compare final answer
                generated = [g.split('####')[0].lstrip(A_DELIM).strip() for g in generated]   

            acc += sum([1 if is_correct(p, gt, task=task) else 0 for gt, p in zip(gt_ans, generated)])
            
        #print("Final ans accuracy: {}".format(acc/len(eval_examples)))
        return {"eval_acc": acc/len(eval_examples), 'generated_solutions': all_generated}
import json, os, math
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import torch as th
import random
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from alignment_utils import StepAligner
from constants import ANS_IDENTIFIER
from torch.nn.utils.rnn import pad_sequence
from data_utils.utils import timeout
import torch


TASK_TO_STEP_DELIMITER = {
    "gsm8k": "|",
    "coin_flip": "|",
    "mathqa": ";",
    "multiarith": "|",
    "svamp": "|",
    "tso": "|",
}
    
class GSMPairwiseRankingDataset(th.utils.data.Dataset):
    '''
    Dataset that returns a pair of positive and negative trajectory suffixes for each question. 
    Examples: Question, Prefix + Positive Step, Prefix + Negative Step
    Only valid for stitching. 
    '''

    def __init__(self, samples, tokenizer, args=None):
        '''
        Args:
            samples: list of dicts with keys: question and values: trajectories, is_correct, gt_sol
            tokenizer: huggingface tokenizer
            max_len: max length of input sequence
        '''
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.step_delimiter = TASK_TO_STEP_DELIMITER[args.task]
        
        if args.step_delimiter is not None:
            print("Overriding step delimiter with {}".format(args.step_delimiter))
            self.step_delimiter = args.step_delimiter
        
        self.args = args
        self.model_style = args.model_style
        self.invalid_prefix_prob = getattr(args, 'invalid_prefix_prob', 0.0)
        self.step_aligner = StepAligner(model=args.step_aligner_model)

        ## if cls and sep are not already added, add them
        if not self.tokenizer.cls_token:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        if not self.tokenizer.sep_token:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        if self.args.task == 'coin_flip' or self.args.task == "tso": ## load nli model
            roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').cuda()
            roberta.eval()  # disable dropout for evaluation
            def is_entailed(a, b):
                tokens = roberta.encode(a, b)
                prediction_1 = roberta.predict('mnli', tokens).argmax().item()
                tokens = roberta.encode(b, a)
                prediction_2 = roberta.predict('mnli', tokens).argmax().item()
                return prediction_1 == 2 and prediction_2 == 2
            self.is_entailed = is_entailed
        
        print("Building pairwise dataset...")
        self.examples = self.step_align_and_get_labels(self.samples, args)


    def _extract_steps(self, sol):
        '''
        Extracts steps from a solution
        '''
        if self.step_delimiter == '. ':
            steps = sent_tokenize(sol) #.split(self.step_delimiter)
        elif self.step_delimiter == '|':
            steps = [s.strip() for s in sol.split(self.step_delimiter) if s.strip()]    
        elif self.step_delimiter == ';':
            steps = [s.strip() for s in sol.split(self.step_delimiter) if s.strip()]
            steps = [s for s in steps if 'print' not in s]
        else:
            raise NotImplementedError(f"step delimiter {self.step_delimiter} not implemented!")

        if self.step_delimiter != '. ': # ince sent_tokenize already keeps the periods
            steps = [s + ' ' + self.step_delimiter if not s.endswith(self.step_delimiter) else s for s in steps]
        
        return steps
         
    def step_align_and_get_labels(self, samples, args):
        '''
        Performs stepwise alignment between sampled trajectories and ground truth ones
        Returns a list of tuples (tokenized question, tokenized answer, stepwise labels)
        '''
        examples = []
        n_skipped = 0
        for sample in tqdm(samples):
            #print("*************************************************************")
            #print("Question: ", sample['question'])
            question = sample['question']
            if not question.startswith('Q: '):
                question = 'Q: ' + question
            
            gt_sol = sample['gt_sol'].replace('.\n', '\n').replace('\n', self.step_delimiter).strip()
            assert self.step_delimiter in gt_sol, f"Step delimiter {self.step_delimiter} not found in {gt_sol}"
            gt_sol = gt_sol.split(ANS_IDENTIFIER)[0]
            gt_sol = "A: " + gt_sol

            is_correct = sample['is_correct']
            
            question_tokens = self.tokenizer(question, padding=False, add_special_tokens=False)["input_ids"]

            if gt_sol.startswith('A: '):
                gt_sol = gt_sol[3:] # remove the 'A: ' prefix before splitting to steps
                gt_sol = ' '.join(gt_sol.split())

            gold_steps = self._extract_steps(gt_sol)
            correct_sols = [gold_steps]

            if args.use_correct_samples: 
                print("Using correct samples for alignment!!")
                ## use correctly sampled solutions for alignment as well
                for i, traj in enumerate(sample['trajectories']):
                    if  is_correct[i]:
                        traj = traj.replace('A:', '').strip()
                        traj = traj.replace('.\n', '\n').replace('\n', self.step_delimiter)
                        traj = ' '.join(traj.split())
                        #assert self.step_delimiter in traj, f"Step delimiter {self.step_delimiter} not found in {traj}"
                        traj = traj.split(ANS_IDENTIFIER)[0]
                        if ANS_IDENTIFIER in traj:
                            traj = traj.split(ANS_IDENTIFIER)[0]
                        traj_steps = self._extract_steps(traj)
                        if not traj_steps:
                            continue
                        correct_sols.append(traj_steps)

            ## go over incorrect solutions and align them with correct ones. 
            for i, traj in enumerate(sample['trajectories']):
                traj = traj.replace('A:', '').strip()
                traj = traj.replace('.\n', '\n').replace('\n', self.step_delimiter)
                traj = ' '.join(traj.split())

                if is_correct[i]:
                    continue

                traj = traj.split(ANS_IDENTIFIER)[0]
                if ANS_IDENTIFIER in traj:
                    traj = traj.split(ANS_IDENTIFIER)[0]

                traj_steps = self._extract_steps(traj)
                
                if not traj_steps:
                    continue
                
                for correct_steps in correct_sols:
                    ## make sure every traj step end with step delimiter
                    assert not traj_steps[0].startswith("A: ") and not correct_steps[0].startswith("A: ") # make sure the first step does not start with 'A: ' to not affect the alignment algorithm
                    if getattr(args, 'skip_alignment', False):
                        aligned_traj = traj_steps
                        aligned_correct = correct_steps
                        if len(aligned_traj) != len(aligned_correct):
                            n_skipped += 1
                            continue
                    else:
                        aligned_traj, aligned_correct, cost = self.step_aligner.compute_alignment_from_trajectories(traj_steps, correct_steps, delimiter=self.step_delimiter)
                        if cost > args.max_alignment_cost: ## roughly three gaps
                            n_skipped += 1
                            continue                        
                    
                    for prefix, positive_step, negative_step in self._get_pairwise_examples_by_stitching_trajectories(aligned_traj, aligned_correct):
                        if not prefix.startswith('A: '):
                            prefix = "A: " + prefix
                        prefix_tokens = self.tokenizer(prefix, padding=False, add_special_tokens=False)["input_ids"]
                        positive_step_tokens = self.tokenizer(positive_step, padding=False, add_special_tokens=False)["input_ids"]
                        negative_step_tokens = self.tokenizer(negative_step, padding=False, add_special_tokens=False)["input_ids"]
                        examples.append((question_tokens, prefix_tokens, positive_step_tokens, negative_step_tokens))

        print("Skipping {} trajectories with high alignment cost > {}".format(n_skipped, args.max_alignment_cost))
        return examples
    
    def _get_pairwise_examples_by_stitching_trajectories(self, aligned_traj, aligned_gt):
        labels = []
        steps = []

        for i, (traj_step, gt_step) in enumerate(zip(aligned_traj, aligned_gt)):
            gt_steps_so_far = [s for s in aligned_gt[:i] if s != '_']

            if traj_step == '_' and gt_step != '_': ## missing step, add gt step to the prefix
                steps.append(gt_step)
                labels.append(1)
                
            elif traj_step != '_' and gt_step == '_': ## extra step
                ### get the next ground truth step that is not '_' to use as positive step
                next_gt_step = [s for s in aligned_gt[i:] if s != '_']
                
                if len(next_gt_step) == 0: ## TODO: MAYBE ADD A STOP TOKEN? 
                    break # no more potential positive steps
                
                next_gt_step = next_gt_step[0]
                yield self._create_pairwise_example(prefix=steps, positive_step=next_gt_step, negative_step=traj_step)
                
                if self.args.break_after_extra_step:
                    break
                
            elif traj_step == '_' and gt_step == '_':
                continue

            else:
                ## two aligned steps
                if not self._is_correct_step(traj_step=traj_step, gt_step=gt_step, prefix=steps):
                    if random.random() < self.invalid_prefix_prob:
                        ## allow invalid prefix
                        steps.append(traj_step)                        
                        labels.append(0)
                        continue
                    else:
                        yield self._create_pairwise_example(prefix=steps, positive_step=gt_step, negative_step=traj_step)

                        if steps != gt_steps_so_far:
                            yield self._create_pairwise_example(prefix=gt_steps_so_far, positive_step=gt_step, negative_step=traj_step)
                        break
                else:
                    steps.append(traj_step)
                    labels.append(1)
    
    def _execute_and_get_var(self, program, var_name):
        env = {}
        with timeout(1, program):
            try:
                exec(program, globals(), env)
            except:
                return None
        return env.get(var_name, None)
    
    def _is_correct_step(self, traj_step, gt_step, prefix=None):
        
        if self.args.task in ['gsm8k', 'svamp', 'multiarith']: # math-natural language tasks
            VAR_RE_EQ = re.compile(r"{}(\-?[0-9\.]+)".format('=')) # = xx 
            VAR_RE_NUM = re.compile(r"\d+[.,]?\d*") # any number in the step (to be used when no equation is present, just a number)
            
            traj_vars = VAR_RE_EQ.findall(traj_step)
            gt_vars = VAR_RE_EQ.findall(gt_step)
            
            if len(traj_vars) == 0:
                #try to find a number in the step instead
                traj_vars = VAR_RE_NUM.findall(traj_step)

            if len(gt_vars) == 0:
                gt_vars = VAR_RE_NUM.findall(gt_step)
            
            if len(traj_vars) == 0 or len(gt_vars) == 0: ## no way to compare the two steps -- so we assume traj is incorrect
                return False # need to run an blation with that correct if len(gt_vars) is 0. 
            
            traj_var = traj_vars[-1] # value not variable
            gt_var = gt_vars[-1]
            try:
                traj_var_f = float(traj_var)
                gt_var_f = float(gt_var)
            except:
                return False
            if abs(traj_var_f - gt_var_f) < 1e-3:
                return True
            else:
                return False
        
        elif self.args.task in ['mathqa', 'asdiv']: ## math-to-code tasks
            ## in the form of VAR = VALUE, VAR is alphaneumeric, VALUE is numeric
            if traj_step.strip() == gt_step.strip():
                return True # exact string, so no need to run the program
        
            VAR_RE = re.compile(r"([a-zA-Z0-9]+)")
            traj_vars = VAR_RE.findall(traj_step)
            gt_vars = VAR_RE.findall(gt_step)

            if len(traj_vars) == 0 or len(gt_vars) == 0:
                return False

            traj_var = traj_vars[0]
            gt_var = gt_vars[0]

            prefix_traj = " ".join(prefix + [traj_step])
            prefix_gt = " ".join(prefix + [gt_step])

            traj_var_val = self._execute_and_get_var(prefix_traj, traj_var)
            gt_var_val = self._execute_and_get_var(prefix_gt, gt_var)

            if traj_var_val is None or gt_var_val is None:
                return False
            try:
                return abs(traj_var_val - gt_var_val) < 1e-4
            except:
                return False
        
        elif self.args.task == 'last_letter_concatenation':
            # find strings between double quotes
            VAR_RE = re.compile(r"\"([a-zA-Z]+)\"")
            traj_vars = VAR_RE.findall(traj_step)
            gt_vars = VAR_RE.findall(gt_step)
            
            if len(traj_vars) == 0 or len(gt_vars) == 0:
                return False
            
            traj_var = traj_vars[-1].strip()
            gt_var = gt_vars[-1].strip()

            if traj_var == gt_var:
                return True

            return False
        
        elif self.args.task == 'coin_flip' or self.args.task == 'tso':
            pred = self.is_entailed(traj_step, gt_step)
            return pred
            #if 'head' in traj_step and 'tail' in gt_step:
            #    return False
            #elif 'tail' in traj_step and 'head' in gt_step:
            #    return False
            
            #return True
        
        else: 
            raise NotImplementedError("is_correct_step not implemented for task {}".format(self.args.task))

    def _create_pairwise_example(self, prefix, positive_step, negative_step):
        #positive_traj = " ".join(prefix  + [positive_step])
        #negative_traj = " ".join(prefix + [negative_step])
        prefix = " ".join(prefix)
        return prefix, positive_step, negative_step
    
    def __len__(self):
        return len(self.examples)
    
    def _process_item_enc_dec(self, qn_tokens, ans_tokens, labels):
        raise NotImplementedError("Not implemented yet")

    def _process_item_enc(self, qn_tokens, prefix_tokens, pos_tokens, neg_tokens):
        '''
        Args:
            qn_tokens: list of token ids for the question
            prefix_tokens: list of token ids for the solution prefix
            pos_tokens: list of token ids for the positive step
            neg_tokens: list of token ids for the negative step
        '''
        ## combine [CLS] + Question + Prefix + [SEP] + Positive/Negative Step
        qn_pos_tokens = [self.tokenizer.cls_token_id] + qn_tokens + prefix_tokens + [self.tokenizer.sep_token_id] + pos_tokens
        qn_neg_tokens = [self.tokenizer.cls_token_id] + qn_tokens + prefix_tokens + [self.tokenizer.sep_token_id] + neg_tokens

        pos_token_type_ids = [0] + [0] * len(qn_tokens + prefix_tokens) + [0] + [1] * len(pos_tokens)
        neg_token_type_ids = [0] + [0] * len(qn_tokens + prefix_tokens) + [0] + [1] * len(neg_tokens)

        assert len(qn_pos_tokens) == len(pos_token_type_ids)
        assert len(qn_neg_tokens) == len(neg_token_type_ids)

        qn_pos_attention_mask = [1] * len(qn_pos_tokens)
        qn_neg_attention_mask = [1] * len(qn_neg_tokens)
        
        if len(qn_pos_tokens) > self.max_len:
            print("trajectory too long, {} vs. {}".format(len(qn_pos_tokens), self.max_len))
            qn_pos_tokens = qn_pos_tokens[:self.max_len]
            pos_token_type_ids = pos_token_type_ids[:self.max_len]
            qn_pos_attention_mask = qn_pos_attention_mask[:self.max_len]
        
        if len(qn_neg_tokens) > self.max_len:
            print("trajectory too long, {} vs. {}".format(len(qn_neg_tokens), self.max_len))
            qn_neg_tokens = qn_neg_tokens[:self.max_len]
            neg_token_type_ids = neg_token_type_ids[:self.max_len]
            qn_neg_attention_mask = qn_neg_attention_mask[:self.max_len]
        
        return dict(pos_input_ids=th.tensor(qn_pos_tokens, dtype=th.long),
                    pos_token_type_ids=th.tensor(pos_token_type_ids, dtype=th.long),
                    neg_input_ids=th.tensor(qn_neg_tokens, dtype=th.long),
                    neg_token_type_ids=th.tensor(neg_token_type_ids, dtype=th.long), 
                    pos_attention_mask=th.tensor(qn_pos_attention_mask, dtype=th.long),
                    neg_attention_mask=th.tensor(qn_neg_attention_mask, dtype=th.long))
    
    def __getitem__(self, idx):
        qn_tokens, prefix_ids, positive_ex_ids, negative_ex_ids = self.examples[idx]
        return self._process_item_enc(qn_tokens, prefix_ids, positive_ex_ids, negative_ex_ids)
        

    def collate_fn(self, batch):
        return self.collate_enc(batch)

    def collate_enc(self, batch):

        pos_input_ids = pad_sequence([x['pos_input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        pos_token_type_ids = pad_sequence([x['pos_token_type_ids'] for x in batch], batch_first=True, padding_value=0)
        neg_input_ids = pad_sequence([x['neg_input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        neg_token_type_ids = pad_sequence([x['neg_token_type_ids'] for x in batch], batch_first=True, padding_value=0)

        pos_attention_mask = pad_sequence([x['pos_attention_mask'] for x in batch], batch_first=True, padding_value=0)
        neg_attention_mask = pad_sequence([x['neg_attention_mask'] for x in batch], batch_first=True, padding_value=0)

        return dict(pos_input_ids=pos_input_ids, pos_token_type_ids=pos_token_type_ids,
                    neg_input_ids=neg_input_ids, neg_token_type_ids=neg_token_type_ids, 
                    pos_attention_mask=pos_attention_mask, neg_attention_mask=neg_attention_mask)
    
    

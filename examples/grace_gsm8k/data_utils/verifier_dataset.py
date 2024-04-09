import torch as th
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from data_utils.utils import timeout


TASK_TO_STEP_DELIMITER = {
    "last_letter_concatenation": "|",
    "gsm8k": "|",
    "mathqa": ";",
    "multiarith": "|",
    "svamp": "|",
    "asdiv": ";",
}

    
class GSMVerifierDataset(th.utils.data.Dataset):
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
        self.args = args
        self.model_style = args.model_style
        self.invalid_prefix_prob = getattr(args, 'invalid_prefix_prob', 0.0)

        ## if cls and sep are not already added, add them
        if not self.tokenizer.cls_token:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        if not self.tokenizer.sep_token:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        
        print("Building verifier dataset...")
        self.examples = self._get_labels(self.samples, args)

        ## percentage of positive and negative examples
        n_positive_examples = [ex for ex in self.examples if ex[2]]
        print("Got %.2f%% positive examples" % (len(n_positive_examples) / len(self.examples) * 100))

         
    def _get_labels(self, samples, args):
        '''
        Performs stepwise alignment between sampled trajectories and ground truth ones
        Returns a list of tuples (tokenized question, tokenized answer, stepwise labels)
        '''
        examples = []
        for sample in tqdm(samples):
            question = sample['question']
            if not question.startswith('Q: '):
                question = 'Q: ' + question
            
            gt_sol = sample['gt_sol'].replace('.\n', '\n').replace('\n', self.step_delimiter).strip()
            
            if not gt_sol.startswith('A:'):
                gt_sol = "A: " + gt_sol

            question_tokens = self.tokenizer(question, padding=False, add_special_tokens=False)["input_ids"]
            gt_sol_tokens = self.tokenizer(gt_sol, padding=False, add_special_tokens=False)["input_ids"]
            examples.append((question_tokens, gt_sol_tokens, 1))
            
            traj_labels = sample['is_correct']
            for traj, lbl in zip(sample['trajectories'], traj_labels):
                #if lbl:
                #    continue # skip positive examples
                if not traj.startswith('A:'):
                    traj = "A: " + traj
                
                if self.step_delimiter not in traj:
                    print("Trajectory {} does not contain step delimiter {}".format(traj, self.step_delimiter))
                    continue # skip trajectories without step delimiter
                
                traj_tokens = self.tokenizer(traj, padding=False, add_special_tokens=False)["input_ids"]
                examples.append((question_tokens, traj_tokens, int(lbl)))
            
            if args.balance:
                ## balance the dataset
                n_positive_examples = [ex for ex in examples if ex[2]]
                n_negative_examples = [ex for ex in examples if not ex[2]]

                # downsample the more frequent class
                if len(n_positive_examples) > len(n_negative_examples):
                    n_positive_examples = n_positive_examples[:len(n_negative_examples)]
                else:
                    n_negative_examples = n_negative_examples[:len(n_positive_examples)]

                examples = n_positive_examples + n_negative_examples

                



        return examples


    def _process_item_enc(self, qn_tokens, traj_tokens, label):
        '''
        Args:
            qn_tokens: tokenized question
            traj_tokens: tokenized trajectory
            label: 1 if correct, 0 if incorrect
        '''
        ## combine [CLS] + Question + [SEP] + Solution
        tokens = [self.tokenizer.cls_token_id] + qn_tokens + [self.tokenizer.sep_token_id] + traj_tokens

        token_type_ids = [0] + [0] * len(qn_tokens) + [0] + [1] * len(traj_tokens)

        assert len(tokens) == len(token_type_ids)
        attention_mask = [1] * len(tokens)
        
        if len(tokens) > self.max_len:
            print("trajectory too long, {} vs. {}".format(len(tokens), self.max_len))
            tokens = tokens[:self.max_len]
            token_type_ids = token_type_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
        
        return dict(input_ids=th.tensor(tokens, dtype=th.long),
                    token_type_ids=th.tensor(token_type_ids, dtype=th.long),
                    attention_mask=th.tensor(attention_mask, dtype=th.long),
                    label=th.tensor(label, dtype=th.long)
        )    
    
    def __getitem__(self, idx):
        tokens, traj_ids, label = self.examples[idx]
        return self._process_item_enc(tokens, traj_ids, label)
        

    def collate_fn(self, batch):
        return self.collate_enc(batch)

    def collate_enc(self, batch):
        input_ids = pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = th.stack([x['label'] for x in batch], dim=0)

        return dict(input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels)
    
    def __len__(self):
        return len(self.examples)

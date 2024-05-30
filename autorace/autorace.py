import os
from typing import Optional, Literal
import time
import pandas as pd
import json
import fire
import jsonlines
from tqdm import tqdm
from openai import OpenAI

# Default settings for Evaluator
MAX_TOKENS = 4096
OPENAI_MODEL = 'gpt-4-1106-preview'
TEMPERATURE = 0.7
TOP_P: float = 1.0
NUM_RETURN_SEQUENCES: int = 1
RATE_LIMIT_PER_MIN: Optional[int] = None
STOP: Optional[str] = None
LOGPROBS: Optional[int] = 0

PROMPT_TYPE_DICT = {
    'gsm8k': 'gsm8k_auto',
    'strategyqa': 'sq_auto',
    'aqua': 'aqua_auto',
    'cosmos': 'cosmos_auto',
    'multistep_arithmetic': 'arith_auto',
    'word_sorting': 'sort_auto',
    'logical_deduction': 'logic_auto'
}

OPENAI_KEY = os.getenv('OPENAI_API_KEY', input('Please input your OpenAI API key: '))
client = OpenAI(
    api_key = OPENAI_KEY
)

def generate(prompt):
    ''' generation using OpenAI API '''
    while(True):
        try:
            # sleep several seconds to avoid rate limit
            if RATE_LIMIT_PER_MIN is not None:
                time.sleep(60 / RATE_LIMIT_PER_MIN)
            if ('gpt-3.5-turbo' in OPENAI_MODEL) or ('gpt-4' in OPENAI_MODEL):
                messages = [{'role': 'user', 'content': prompt}]
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    n=NUM_RETURN_SEQUENCES,
                    stop=STOP,
                )
                text=[choice.message.content for choice in response.choices]
                return text
            else:
                raise ValueError(f'Unknown OPENAI MODEL {OPENAI_MODEL}')
        except Exception as e:
            print(f'An Error Occured: {e}, sleeping for 5 seconds')
            time.sleep(5)

def autorace_criterion(dataset:str = 'aqua', example_wrong_chains:str = 'EXAMPLE_WRONG_CHAINS_AQUA.txt'):
    
    '''
    This function is used to generate criterions by comparing reference/student answers. (Fig 2 in the paper).
    
        
    As shown in Fig 2, we should provide several wrong reasoning chains to generate criterions.
    
    We provide an example in EXAMPLE_WRONG_CHAINS_AQUA.txt, which includes several wrong reasoning chains on the AQuA dataset.

    Please follow the format in EXAMPLE_WRONG_CHAINS_AQUA.txt. i.e.
    -----------------------------------------------------------------------------
    Question:
    The original price of an item is discounted 22%. A customer buys the item at this discounted price using a $20-off coupon. There is no tax on the item, and this was the only item the customer bought. If the customer paid $1.90 more than half the original price of the item, what was the original price of the item? Options: A)$61, B)$65, C)$67.40, D)$70, E)$78.20

    Reference answer:
    Let x be the original price of the item
    Discounted price = 0.78x
    Payment made by the customer after using the $20 coupon = 0.78x - 20
    0.78x - 20 = x/2 + 1.9
    x = 78.20
    Answer: E

    Student answer:
    The original price of the item is 1.22 * $20. The answer is B.
    -----------------------------------------------------------------------------
    Please see `EXAMPLE_WRONG_CHAINS_AQUA.txt` for details. 
    '''
    
    assert os.path.exists(example_wrong_chains), f'example_wrong_chains: {example_wrong_chains} does not exist!'
    
    with open(example_wrong_chains) as f:
        EXAMPLE_WRONG_CHAINS = f.read()

    with open('prompt.json') as f:
        prompt = json.load(f)
        
    if f"{dataset}_auto" in prompt:
        print(f'Warning: dataset {dataset} already exists in prompt.json, please check whether you want to overwrite it.')
        input('Press any key to continue...')
        
    criterion_prompt = prompt['criterion'].format(EXAMPLE_WRONG_CHAINS)
    #prompt 'criterion' is used for generating criterions
    criterion_text = generate(criterion_prompt)
    print(criterion_text)
    criterion = '1. **' + criterion_text[0].split('1. **')[-1]
    #user need to human check this criterion cuz it's not cleaned enough
    import re
    criterion = re.sub(r'\d\. ', '', criterion)
    evaluation_prompt = 'Below is a question and an answer from a student. You are required to check the correctness of the reasoning chains step by step. The criterions are as follows:\n\n{}\n\nQuestion:\n{{}}\n\nStudent answer:\n{{}}\n\nPlease check the answer through each criterion, and make sure you carefully examine each reasoning step. Finally, if there is any step that fails the verification, output a INCORRECT, else output a CORRECT.'.format(criterion)
    prompt[dataset + '_auto'] = evaluation_prompt

    with open('prompt.json', 'w') as f:
        json.dump(prompt, f)

def autorace_score(output_log_path:str):
    '''report autorace score'''
    #load the autorace evaluation
    with jsonlines.open(output_log_path, mode='r') as reader:
        autorace = list(reader)

    #calculate the score
    total = len(autorace)
    incorrect = 0
    for i in range(total):
        if 'INCORRECT' in autorace[i]['evaluation_result'][0]:
            incorrect += 1

    print(f'autorace score: {(total - incorrect) / total:.2f}')

def autorace_evaluation(
    dataset: str = "gsm8k", 
    reasoning_model: str = "eval_model",
    output_log_dir:str = 'logs/auto_race'
):
    '''
    autorace evaluation, 
    - In Tab.1, we use this function to first generate the evaluation results, then use 'test_evaluation_accuracy' to compare with the human label, and finally get the Table 1 results.
    - In Tab.9, we use this function to calculate autorace score for gsm8k, aqua, strategyqa.
    
    specify the dataset and reasoning_model to evaluate.
    '''
    
    predefined_datasets = ['gsm8k', 'strategyqa', 'aqua', 'cosmos', 'multistep_arithmetic', 'word_sorting', 'logical_deduction']
    
    if dataset not in predefined_datasets:
        print(f"Warning: The dataset '{dataset}' is not a predefined dataset.")
    if dataset not in PROMPT_TYPE_DICT:
        raise ValueError(f"dataset '{dataset}' is not in PROMPT_TYPE_DICT! Please add the prompt type to PROMPT_TYPE_DICT.")
    
    
    data_path = f'./data/{reasoning_model}/{dataset}.jsonl'
    assert os.path.exists(data_path), f'the output from {reasoning_model}: {data_path} does not exist!'
    
    #specify a log dir
    assert output_log_dir is not None, 'output_log_dir should not be None'
    output_log_dir = os.path.join(output_log_dir, reasoning_model, dataset)
    os.makedirs(output_log_dir, exist_ok=True)
    output_log_path = f'{output_log_dir}/autorace_eval.jsonl'
    
    print("evaluating reasoning model: ", reasoning_model, " on dataset: ", dataset, "output log path: ", output_log_path)
    
    # generate evaluator's response
    import pandas as pd
    data = pd.read_json(data_path, lines=True)
    results = []
    for index in tqdm(range(len(data))):
        reasoning_chain = data.loc[index, 'reasoning_chain']
        #make some format cleanning
        if not reasoning_chain.startswith('\n'):
            reasoning_chain = '\n' + reasoning_chain#new line at the beginning
        reasoning_chain = reasoning_chain.rstrip('\n\n.')
        raw_question = data.loc[index, 'question']
        raw_question = raw_question.replace('Q:', '')
        raw_question = raw_question.lstrip(' ')
        with open('prompt.json') as f:
            prompts = json.load(f)
        prompt = prompts[PROMPT_TYPE_DICT[dataset]].format(raw_question, reasoning_chain)  
        prompt = prompt.replace('..', '.')
        evaluation_result = generate(prompt)
        tmp = {'index': index, 'evaluation_result': evaluation_result, 'question': raw_question, 'reasoning_chain': reasoning_chain, 'answer': data.loc[index, 'answer'], 'prompt': prompt}    
        results.append(tmp)
        with jsonlines.open(output_log_path, mode='w') as writer:
            writer.write_all(results)
    
    #calculate the score
    autorace_score(output_log_path)
    

def test_evaluation_accuracy(output_name: str = time.strftime('%Y-%m-%d-%H-%M-%S')):
    """
    This function is used to test the accuracy of evaluation metrics, when using human labels as the ground truth.
    (Reproduce Tab.1 in the paper)
    
    Args:
        output_name (_type_, optional): Name for the output directory. Defaults to current timestamp.
    """
    
    print("Start testing evaluation accuracy...")
    
    datasets = ['gsm8k','strategyqa','cosmos', 'multistep_arithmetic','word_sorting','logical_deduction']

    model = "eval_model"
    eval_dir = "./logs/auto_race"
    human_label_dir = "./data/eval_model"
    
    for dataset in datasets:
        if os.path.exists(f'{eval_dir}/{model}/{dataset}'):
            print(f'{eval_dir}/{model}/{dataset} exists, pass.')
        else:
            print(f'{eval_dir}/{model}/{dataset} does not exist, start autorace evaluation...')
            autorace_evaluation(dataset, model, eval_dir)
    
        human_label_path = os.path.join(human_label_dir, f'{dataset}.jsonl')
        evaluator_label_path = os.path.join(eval_dir, f'{model}/{dataset}/autorace_eval.jsonl')
        
        with jsonlines.open(human_label_path, mode='r') as reader:
            human_labels = list(reader)
        
        with jsonlines.open(evaluator_label_path, mode='r') as reader:
            evaluator_labels = list(reader)
            
        assert len(human_labels) >= len(evaluator_labels), f'there are unlabelled samples in {human_label_path} compared to {evaluator_label_path}!'

        total = len(evaluator_labels)
        score = 0
        correct_align_list = []
        incorrect_align_list = []
        incorrect_disagreement = []
        correct_disagreement = []
        
        for i in range(len(evaluator_labels)):
            output = evaluator_labels[i]['evaluation_result'][0]
            if 'INCORRECT' in output:
                if int(human_labels[i]['human_label']) == 0:
                    incorrect_align_list.append(i)
                    score += 1
                else:
                    incorrect_disagreement.append({
                        'index': i, 
                        'prompt': evaluator_labels[i]['prompt'], 
                        'answer': str(human_labels[i]['answer']), 
                        'human_label': str(human_labels[i]['human_label']), 
                        'evaluation_result': evaluator_labels[i]['evaluation_result']
                    })
            else:
                if int(human_labels[i]['human_label']) == 1:
                    correct_align_list.append(i)
                    score += 1
                else:
                    correct_disagreement.append({
                        'index': i, 
                        'prompt': evaluator_labels[i]['prompt'], 
                        'answer': str(human_labels[i]['answer']), 
                        'human_label': str(human_labels[i]['human_label']), 
                        'evaluation_result': evaluator_labels[i]['evaluation_result']
                    })

        output_dir = f'logs/error_analysis/{output_name}/{dataset}'
        correct_path = os.path.join(output_dir, 'correct_disagree')
        incorrect_path = os.path.join(output_dir, 'incorrect_disagree')
        align_score_log = os.path.join(output_dir, 'align_score.txt')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(correct_path, exist_ok=True)
        os.makedirs(incorrect_path, exist_ok=True)

        for sample in incorrect_disagreement:
            with open(os.path.join(incorrect_path, f"{sample['index']}.txt"), 'w') as f:
                f.write('====================================\n')
                f.write(f'Index: {sample["index"]}\n')
                f.write(f'Answer: {sample["answer"]}\n')
                f.write(f'Human label: {sample["human_label"]}\n')
                f.write('====================================\n')
                f.write(f'Prompt: {sample["prompt"]}\n')
                f.write('====================================\n')
                f.write(f'Evaluation: {sample["evaluation_result"]}\n')
                
        for sample in correct_disagreement:
            with open(os.path.join(correct_path, f"{sample['index']}.txt"), 'w') as f:
                f.write('====================================\n')
                f.write(f'Index: {sample["index"]}\n')
                f.write(f'Answer: {sample["answer"]}\n')
                f.write(f'Human label: {sample["human_label"]}\n')
                f.write('====================================\n')
                f.write(f'Prompt: {sample["prompt"]}\n')
                f.write('====================================\n')
                f.write(f'Evaluation: {sample["evaluation_result"]}\n')

        align_score = score / total
        print(f'align score for {dataset}: {align_score:.2f}')
        with open(align_score_log, 'w') as f:
            f.write(f'Align score: {align_score:.2f}\n')
            f.write(f'Total: {total}\n')
            f.write(f'Correct: {score}\n')
            f.write(f'Incorrect: {total - score}\n')
            f.write(f'Correct align list: {correct_align_list}\n')
            f.write(f'Incorrect align list: {incorrect_align_list}\n')
            f.write(f'Correct disagreement list: {correct_disagreement}\n')
            f.write(f'Incorrect disagreement list: {incorrect_disagreement}\n')    
    

def main(gen_criteria: bool = False, dataset: str = 'gsm8k', example_wrong_chains: str = 'EXAMPLE_WRONG_CHAINS_AQUA.txt',  reproduce_tab1: bool = False, reasoning_model: str = "eval_model", output_log: str = 'logs/auto_race'):
    if reproduce_tab1:
        test_evaluation_accuracy()
    elif gen_criteria:
        autorace_criterion(dataset, example_wrong_chains)
    else:
        autorace_evaluation(dataset, reasoning_model, output_log)
    

if __name__ == '__main__':
    
    fire.Fire(main)
    


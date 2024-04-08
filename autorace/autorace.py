import os
from typing import Optional, Literal
import time
import pandas as pd
import json
import fire
import jsonlines
from scipy.stats import somersd
from tqdm import tqdm
from openai import OpenAI
max_tokens = 4096
model = "gpt-4-1106-preview"
temperature = 0.7
top_p: float = 1.0
num_return_sequences: int = 1
rate_limit_per_min: Optional[int] = None
stop: Optional[str] = None
logprobs: Optional[int] = 0

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY", None)
)

def generate(prompt):
    while(True):
        try:
            # sleep several seconds to avoid rate limit
            if rate_limit_per_min is not None:
                time.sleep(60 / rate_limit_per_min)
            if ('gpt-3.5-turbo' in model) or ('gpt-4' in model):
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop,
                )
                text=[choice.message.content for choice in response.choices]
                return text
            else:
                raise ValueError(f"Unknown model {model}")
        except Exception as e:
            print(f"An Error Occured: {e}, sleeping for 5 seconds")
            time.sleep(5)

def AutoRace_evaluation(prompt_type:str = "aqua_auto",
                    output_log:str = "logs/example_AutoRace.json",
                    data_pth = "./eval_example.json"
                    ):
    annotated_data = pd.read_json(data_pth, orient='records')
    for index in tqdm(range(len(annotated_data))):
        metadata_generation = annotated_data.loc[index, 'cot']
        #make some format cleanning
        metadata_generation = '\n' + metadata_generation
        metadata_generation = metadata_generation.rstrip('\n\n.')
        raw_question = annotated_data.loc[index, 'question']
        raw_question = raw_question.replace('Q:', '')
        raw_question = raw_question.lstrip(' ')

        with open("prompt.json") as f:
            prompt = json.load(f)
        prompt = prompt[prompt_type].format(raw_question, metadata_generation)  
        prompt = prompt.replace('..', '.')
        text = generate(prompt)
        tmp = {'index': index, 'text': text, 'question': raw_question, 'metadata_generation': metadata_generation}    
        with jsonlines.open(output_log, mode='a') as writer:
            writer.write(tmp)
    
def AutoRace_criterion(task_type:str = "aquatest"):
    #example 4 shot for AQuA-RAT, user can change to any other task's prompt
    few_shot_prompt = """
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

Question:
20 marbles were pulled out of a bag of only white marbles, painted black, and then put back in. Then, another 20 marbles were pulled out, of which 1 was black, after which they were all returned to the bag. If the percentage of black marbles pulled out the second time represents their percentage in the bag, how many marbles in total Q does the bag currently hold? Options: A)40, B)200, C)380, D)400, E)3200

Reference answer:
We know that there are 20 black marbles in the bag and this number represent 1/20 th of the number of all marbles in the bag, thus there are total Q of 20*20=400 marbles.
Answer: D.

Student answer:
The probability that the first marble is white is 19/20. The probability that the second marble is white is 18/19. The probability that the third marble is white is 17/18. In total, the probability that the pulled out marbles are all white is 19/20 * 18/19 * 17/18 = 1167/1360. The probability that at least one marble is black is 1 - 1167/1360 = 193/1360. The percentage of black marbles is 193/1360 * 100% = 14.19%. The number of black marbles is 14.19% * 640 = 90. Therefore, there are 640 marbles in total. The answer is D.

Question:
Of the 200 students in a school, at least 45% attended the prom night and at least 35% took part in the debating session. What is the maximum number of students who could have neither attended the prom night nor the debating session? Options: A)27, B)81, C)90, D)99, E)110

Reference answer:
To maximize the number of students who did neither, we should minimize the number of students who debated or attended the prom.
Let's assume that all 35% of students who debated also attended the prom.
Then 35% did both, 10% only attended prom, and 55% did neither.
0.55*200 = 110
The answer is E.

Student answer:
The minimum number of students who attended the prom night is 45% * 200 - 45% * 150 = 45 - 35 = 10. The minimum number of students who took part in the debating session is 35% * 200 - 35% * 150 = 35 - 20 = 15. The maximum number of students who neither attended the prom night or the debating session is 200 - 10 - 15 = 175. The answer is C.

Question:
A company produces 420 units of a particular computer component every month, at a production cost to the company of $110 per component, and sells all of the components by the end of each month. What is the minimum selling price per component that will guarantee that the yearly profit (revenue from sales minus production costs) will be at least $626,400 ? Options: A)226, B)230, C)240, D)260, E)280

Reference answer:
450*12(x-110)=626400
where x is a selling cost of one item
x-110, is a profit from one item
450 - number of items produced and sold per month
12 - is a number of month in a year
Simplifying the equation will lead to x-110=116, then x = 230
B

Student answer:
The profit per component is (selling price - production cost) = $s - $110. The yearly profit is $y = n * (selling price - production cost) = 420 * 12 * ($s - $110). $y is at least $626,400. This is $y \u2265 $626,400. The selling price s must be s \u2265 240 to guarantee that the yearly profit is at least $626,400. The answer is C.
"""
    with open("prompt.json") as f:
        prompt = json.load(f)
    criterion_prompt = prompt["criterion"].format(few_shot_prompt)
    #prompt 'criterion' is used for generating criterions
    criterion_text = generate(criterion_prompt)
    print(criterion_text)
    criterion = '1. **' + criterion_text[0].split('1. **')[-1]
    #user need to human check this criterion cuz it's not cleaned enough
    import re
    criterion = re.sub(r'\d\. ', '', criterion)
    evaluation_prompt = "Below is a question and an answer from a student. You are required to check the correctness of the reasoning chains step by step. The criterions are as follows:\n\n{}\n\nQuestion:\n{{}}\n\nStudent answer:\n{{}}\n\nPlease check the answer through each criterion, and make sure you carefully examine each reasoning step. Finally, if there is any step that fails the verification, output a INCORRECT, else output a CORRECT.".format(criterion)
    prompt[task_type + '_auto'] = evaluation_prompt

    with open("prompt.json", "w") as f:
        json.dump(prompt, f)

def result_score(data:pd.DataFrame, output_log_dir:str):
    #load the AutoRace evaluation
    with jsonlines.open(output_log_dir, mode='r') as reader:
        autorace = list(reader)

    #calculate the score
    total = len(data)
    incorrect = 0
    for i in range(total):
        if "INCORRECT" in autorace[i]['text'][0]:
            incorrect += 1

    print(f"AutoRace score: {(total - incorrect) / total:.2f}")



def AutoRace_eval_dataset(
    dataset: Literal['gsm8k','strategyqa','AQuA','cosmos', 'multistep_arithmetic','word_sorting','logical_deduction'], 
    model: Literal['dbrx','gpt-4-turbo','claude-3-opus','gemini-pro','internlm-2-7b','llama-2-70b','qwen-1.5-7b','gemma-7b','mistral-7b','llama-2-13b'],
    prompt_type: Literal['gsm8k_auto','sq_auto','cosmos_auto', 'aqua_auto', 'arith_auto','sort_auto','logic_auto'],
    output_log_dir:str = "logs/AutoRace"
):
    #specify a log dir
    import time
    if output_log_dir == "logs/AutoRace":
        output_log_dir = f"logs/{dataset}_{model}"
    os.makedirs(output_log_dir, exist_ok=True)
    output_log_dir = f"{output_log_dir}/{time.strftime('%Y-%m-%d-%H-%M-%S')}.jsonl"
    
    #generate LLM's response
    import pandas as pd
    data = pd.read_json(f"./data/{dataset}_{model}.jsonl", lines=True)
    for index in tqdm(range(len(data))):
        metadata_generation = data.loc[index, 'metadata_generation']
        #make some format cleanning
        if not metadata_generation.startswith('\n'):
            metadata_generation = '\n' + metadata_generation#new line at the beginning
        metadata_generation = metadata_generation.rstrip('\n\n.')
        raw_question = data.loc[index, 'question']
        raw_question = raw_question.replace('Q:', '')
        raw_question = raw_question.lstrip(' ')
        with open("prompt.json") as f:
            prompt = json.load(f)
        prompt = prompt[prompt_type].format(raw_question, metadata_generation)  
        prompt = prompt.replace('..', '.')
        text = generate(prompt)
        tmp = {'index': index, 'text': text, 'question': raw_question, 'metadata_generation': metadata_generation, 'gt_answer': data.loc[index, 'gt_answer'], 'human_label': data.loc[index, 'human_label']}    
        with jsonlines.open(output_log_dir, mode='a') as writer:
            writer.write(tmp)
    

    #calculate the score
    result_score(data, output_log_dir)
    

    
    

if __name__ == '__main__':
    # fire.Fire(AutoRace_evaluation)
    # fire.Fire(AutoRace_criterion)
    fire.Fire(AutoRace_eval_dataset)


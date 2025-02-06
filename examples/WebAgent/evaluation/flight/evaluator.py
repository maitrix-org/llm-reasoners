import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from datetime import datetime

import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..', '..', '..', '..'))
from reasoners.agent.llm import parser

# Get the directory of the current file
default_api_key_path = os.path.join(current_directory, '..', '..', 'default_api_key.txt')
if os.path.exists(default_api_key_path):
    DEFAULT_API_KEY = open(default_api_key_path).read().strip()
else:
    DEFAULT_API_KEY = os.environ.get('OPENAI_API_KEY', None)
    
def retry(client, messages, keys, num_retries=5): 
    tries = 0
    for i in range(num_retries):
        completion = client.chat.completions.create(
            model='gpt-4o', messages=messages
        )
        response = completion.choices[0].message.content
        ans_dict, success, error = parser(response, keys)
        if success:
            return ans_dict
        
        tries += 1
        messages.append({'role': 'assistant', 'content': response})
        msg = f'Query failed. Retrying {tries}/{num_retries}.\n[LLM]:\n{response}\n[User]:\n{error}'
        print(msg)
        messages.append({'role': 'user', 'content': error})
    raise ValueError(f'Could not parse a valid value after {num_retries} retries.')

class FlightSearchEvaluator:
    def __init__(self, questions_path, start_idx=0, end_idx=99999, api_key=None):
        if not api_key:
            api_key = DEFAULT_API_KEY
        print('Using API Key:', api_key)
        self.client = OpenAI(api_key=api_key)
        
        # questions_path = '../task_data/flight_search_questions_no_pass_rel_date_20.csv'
        df = pd.read_csv(questions_path)
        df = df.iloc[start_idx:end_idx]
        self.question_to_constraints_dict = df.set_index('question').to_dict(orient='index')
        
    def evaluate(self, browsing_data_path):
        data = json.load(open(browsing_data_path))
        goal = data['goal']
        if goal not in self.question_to_constraints_dict:
            print(f'Skipping {browsing_data_path} due to not being included in evaluated questions.')
            return None
        constraints = self.question_to_constraints_dict[goal]['constraints']
        
        data_datetime_str = os.path.basename(browsing_data_path).split('.')[0].split('_')[-1]
        data_datetime = datetime.strptime(data_datetime_str, '%Y-%m-%d-%H-%M-%S')
        data_datetime_prompt = data_datetime.strftime('%a, %b %d, %Y %H:%M:%S')
        
        if not data['history']:
            return {'instance': os.path.basename(browsing_data_path),
                    'goal': goal, 
                    'constraints': constraints, 
                    'observations': [],
                    'message': None, 
                    'outcome': 'No Output', 
                    'grounded': False, 
                    'relevant': False,
                    'llm_output': None}
        
        observations = []
        for step in data['history']:
            if 'obs_info' in step[-1]:
                observation = step[-1]['obs_info']
            elif 'observation' in step[-1]:
                observation = step[-1]['observation']
            else:
                raise RuntimeError("No observation info found.")
            obs = observation['clean_axtree_txt']
            observations.append(obs)
            
        last_action = data['history'][-1][-1]['action']
        outcome = 'Response Returned'
        message = None
        error = data.get('error')

        final_action = data['history'][-1][1]
        if final_action.startswith('send_msg_to_user'): 
            start_idx = len('send_msg_to_user(')
            end_idx = len(final_action) - len(')')
            try:
                message = eval(final_action[start_idx:end_idx])
            except SyntaxError:
                outcome = 'Action Error'
            else:
                if message == 'Error encountered when browsing.':
                    outcome = 'Webpage Parsing Error'
                elif message == 'Too many errors encountered. Task failed.':
                    outcome = 'Action Error'
                elif message == "Repetitive actions. Ending the task.":
                    outcome = 'Repetitive Actions'
        elif error:
            outcome = 'Browser Crashed'
        else: 
            outcome = 'Max Steps Reached'
            
        # if outcome == 'Response Returned' and False:
        if outcome == 'Response Returned':
            step_template = "## Step {step_num} Observation:\n\n{obs}"
            step_prompts = [step_template.format(step_num=i+1, obs=obs) for i, obs in enumerate(observations)]
            history_prompt = '\n\n'.join(step_prompts)
            
            prompt = self._get_evaluation_prompt(data_datetime_prompt, history_prompt, constraints, goal, message)
            messages = [
                {'role': 'user', 'content': prompt},
            ]
            
            ans_dict = retry(self.client, messages, 
                             ['think', 'grounding', 'relevance'])
            
            # grounded = ans_dict['grounding'].lower() == 'yes'
            grounded = 'yes' in ans_dict['grounding'].lower()
            # relevant = ans_dict['relevance'].lower() == 'yes'
            relevant = 'yes' in ans_dict['relevance'].lower()
        
        else: 
            ans_dict = None
            grounded = False
            relevant = False
        
        return {'instance': os.path.basename(browsing_data_path),
                'goal': goal, 
                'constraints': constraints, 
                'observations': observations,
                'message': message, 
                'outcome': outcome, 
                'grounded': grounded, 
                'relevant': relevant,
                'llm_output': ans_dict}

    def _get_evaluation_prompt(self, datetime_prompt, history_prompt, constraints, goal, message):
        prompt = f"""\
# Interaction Date and Time:

{datetime_prompt}

# Interaction History:

{history_prompt}

Above are the webpages an assistant interacted with while trying to answer the user's query.

The user is looking for flights with the following constraints:

{constraints}

Here is the exact query provided by the user:

{goal}

Here is the assistant's response: 

{message}

Your task is to evaluate two aspects of the response: 

1) Whether the assistant's response is supported by the interaction history, and 
2) Whether the assistant's response satisfies the user constraints to the extent allowed by the results.

Some Context:

- To determine the seating class of a flight being returned, refer to the value of the "Change seating class" combobox.
- It is not always possible to satisfy all the user constraints. In this case, examine whether the response is as close to the user constraints as possible.

Answer in the following format:

<think>
Your thoughts and reasoning.
</think>

<grounding>
Your assessment of whether the response is supported by the interaction history. Answer "yes" or "no"
</grounding>

<relevance>
Your assessment of whether the response satisfies the user constraints to the extent allowed by the results. Answer "yes" or "no"
</relevance>
"""
        return prompt
import os
import random

import yaml
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
np.random.seed(42)
import copy
import time
from tqdm import tqdm
class ResponseGenerator:
    def __init__(self, config_file, engine, verbose, ignore_existing):
        self.engine = engine
        self.verbose = verbose
        self.ignore_existing = ignore_existing
        self.max_gpt_response_length = 500
        self.data = self.read_config(config_file)
        if self.engine == 'bloom':
            self.model = self.get_bloom()
        elif 'finetuned' in self.engine:
            # print(self.engine)
            assert self.engine.split(':')[1] is not None
            model = ':'.join(self.engine.split(':')[1:])
            # print(model)
            self.engine='finetuned'
            self.model = {'model':model}
        else:
            self.model = None
    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    def get_bloom(self):
        max_memory_mapping = {0: "0GB", 1: "43GB", 2: "43GB", 3: "43GB", 4: "43GB", 5: "43GB"}
        cache_dir = os.getenv('BLOOM_CACHE_DIR', '/data/karthik/LLM_models/bloom/')
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", cache_dir=cache_dir,
                                                     local_files_only=False, load_in_8bit=True, device_map='auto',
                                                     max_memory=max_memory_mapping)
        return {'model': model, 'tokenizer': tokenizer}

    def get_responses(self, task_name, specified_instances = [], run_till_completion=False):
        output_dir = f"responses/{self.data['domain_name']}/{self.engine}/"
        os.makedirs(output_dir, exist_ok=True)
        output_json = output_dir+f"{task_name}.json"
        while True:
            if os.path.exists(output_json):
                with open(output_json, 'r') as file:
                    structured_output = json.load(file)
            else:
                prompt_dir = f"prompts/{self.data['domain_name']}/"
                assert os.path.exists(prompt_dir+f"{task_name}.json")
                with open(prompt_dir+f"{task_name}.json", 'r') as file:
                    structured_output = json.load(file)
                structured_output['engine'] = self.engine        
        
            failed_instances = []
            for instance in tqdm(structured_output["instances"]):
                if "llm_raw_response" in instance:
                    if instance["llm_raw_response"] and not self.ignore_existing:
                        if self.verbose:
                            print(f"Instance {instance['instance_id']} already completed")
                        continue
                if len(specified_instances) > 0:
                    if instance['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance['instance_id'])                   
                
                if self.verbose:
                    print(f"Sending query to LLM: Instance {instance['instance_id']}")
                query = instance["query"]
                stop_statement = "[STATEMENT]"
                if 'caesar' in self.data['domain_name']:
                    stop_statement = caesar_encode(stop_statement)
                llm_response = send_query(query, self.engine, self.max_gpt_response_length, model=self.model, stop=stop_statement)
                if not llm_response:
                    failed_instances.append(instance['instance_id'])
                    print(f"Failed instance: {instance['instance_id']}")
                    continue
                if self.verbose:
                    print(f"LLM response: {llm_response}")
                instance["llm_raw_response"] = llm_response
                with open(output_json, 'w') as file:
                    json.dump(structured_output, file, indent=4)
            
            if run_till_completion:
                if len(failed_instances) == 0:
                    break
                else:
                    print(f"Retrying failed instances: {failed_instances}")
                    time.sleep(5)
            else:
                break
        
if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to run \
    \n t1 = Plan Generation\
    \n t1_zero = Zero Shot Plan Generation\
    \n t1_cot = Plan Generation COT\
    \n t1_pddl = Plan Generation PDDL\
    \n t1_zero_pddl = Zero Shot Plan Generation PDDL\
    ')
    parser.add_argument('--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n bloom = Bloom \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        \n davinci = GPT-3 Davinci \
                        \n curie = GPT-3 Curie \
                        \n babbage = GPT-3 Babbage \
                        \n ada = GPT-3 Ada \
                        ')
                        
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    #config
    parser.add_argument('--config', type=str, required=True, help='Config file name (no need to add .yaml)')
    parser.add_argument('--run_till_completion', type=str, default="False", help='Run till completion')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    # parser.add_argument('--random_example', type=str, default="False", help='Random example')
    args = parser.parse_args()
    task = args.task
    engine = args.engine
    config = args.config
    specified_instances = args.specific_instances
    verbose = eval(args.verbose)
    run_till_completion = eval(args.run_till_completion)
    ignore_existing = args.ignore_existing
    print(f"Task: {task}, Engine: {engine}, Config: {config}, Verbose: {verbose}, Run till completion: {run_till_completion}")
    # specified_instances = args.specified_instances
    # random_example = eval(args.random_example)
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'
    response_generator = ResponseGenerator(config_file, engine, verbose, ignore_existing)
    task_dict = {
        't1': 'task_1_plan_generation',
        't1_zero': 'task_1_plan_generation_zero_shot',
        't1_cot': 'task_1_plan_generation_state_tracking',
        't1_pddl': 'task_1_plan_generation_pddl',
        't1_zero_pddl': 'task_1_plan_generation_zero_shot_pddl',
    }
    try:
        task_name = task_dict[task]
    except:
        raise ValueError("Invalid task name")
    response_generator.get_responses(task_name, specified_instances, run_till_completion=run_till_completion)
   





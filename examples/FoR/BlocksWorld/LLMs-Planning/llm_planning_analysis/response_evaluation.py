import os
import random

import yaml
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
from model_parser.writer_new import ModelWriter
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import json
np.random.seed(42)
from tqdm import tqdm
class ResponseEvaluator:
    def __init__(self, config_file, engine, specified_instances, verbose, ignore_existing=False):
        self.engine = engine
        self.verbose = verbose
        self.ignore_existing = ignore_existing
        self.specified_instances = specified_instances
        self.data = self.read_config(config_file)
        self.instance_dir = self.data['instance_dir']
        self.domain_pddl = f'./instances/{self.data["domain_file"]}'
        self.llm_plan_file = 'llm_plan'
        self._set_task_params()

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
        
    def _set_task_params(self, instance_dir=None):
        if instance_dir is None:
            instance_dir = self.instance_dir
        self.instance_folder = f'./instances/{instance_dir}/'
        self.instance = f'./instances/{instance_dir}/{self.data["instances_template"]}'
        self.n_files = min(self.data['n_instances'], len(os.listdir(self.instance_folder)))

        self.i_start = self.data['start']
        self.i_end = self.data['end']
    
    def get_problem(self, instance, domain):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        return reader.parse_instance(instance)

    def get_executor(self, instance, domain, ground=False):
        plan_executor = Executor(domain, instance, ground=ground)
        return plan_executor
    
    def write_new_instance(self, new_model):
        writer = ModelWriter(new_model)
        writer.write_files('pr-new-domain.pddl', 'pr-new-problem.pddl')

    def load_json(self, task_name):
        response_dir = f"responses/{self.data['domain_name']}/{self.engine}/"        
        output_dir = f"results/{self.data['domain_name']}/{self.engine}/"
        if not self.ignore_existing and os.path.exists(output_dir+f"{task_name}.json"):
            load_dir = output_dir
        else:
            assert os.path.exists(response_dir+f"{task_name}.json")
            load_dir = response_dir
        with open(load_dir+f"{task_name}.json", 'r') as file:
            structured_output = json.load(file)
        return structured_output
            
    def save_json(self, structured_output, task_name):
        output_dir = f"results/{self.data['domain_name']}/{self.engine}/"        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_dir+f"{task_name}.json", 'w') as file:
            json.dump(structured_output, file, indent=4)

    def evaluate_plan(self, task_name):
        structured_output = self.load_json(task_name)
        total_correct = 0
        total_instances = 0
        if 'plan_generalization' in task_name:
            self._set_task_params(instance_dir=self.data['generalized_instance_dir'])
        for instance_dict in tqdm(structured_output["instances"]):
            if "llm_raw_response" in instance_dict:
                if not instance_dict["llm_raw_response"]:
                    if self.verbose:
                        print(f"Instance {instance_dict['instance_id']} response not generated")
                    continue
                if len(self.specified_instances) > 0:
                    if instance_dict['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance_dict['instance_id'])      
                
                if self.verbose:
                    print(f"Evaluting instance {instance_dict['instance_id']}")
                llm_response = instance_dict["llm_raw_response"]
                id = instance_dict["instance_id"]
                cur_instance = self.instance.format(id)
                problem = self.get_problem(cur_instance, self.domain_pddl)
                plan_executor = self.get_executor(cur_instance, self.domain_pddl)
                try:
                    llm_plan, _ = text_to_plan(llm_response, problem.actions, self.llm_plan_file, self.data)
                    instance_dict["extracted_llm_plan"] = llm_plan
                    
                    correct = int(validate_plan(self.domain_pddl, cur_instance, self.llm_plan_file))
                except:
                    correct = int(False)
                    print(f"Warning: Plan extraction failed for instance {id}")    
                if self.verbose:
                    print(f"Correct: {bool(correct)}")
                instance_dict["llm_correct"] = bool(correct)
                total_correct += correct
                total_instances += 1
                self.save_json(structured_output, task_name)
        if self.verbose:
            print(f"Total correct: {total_correct}")
            print(f"Total instances: {total_instances}")
            print(f"Accuracy: {total_correct/total_instances}")
    
    def evaluate_plan_pddl(self, task_name):
        structured_output = self.load_json(task_name)
        total_correct = 0
        total_instances = 0
        if 'plan_generalization' in task_name:
            self._set_task_params(instance_dir=self.data['generalized_instance_dir'])
        for instance_dict in tqdm(structured_output["instances"]):
            if "llm_raw_response" in instance_dict:
                if not instance_dict["llm_raw_response"]:
                    if self.verbose:
                        print(f"Instance {instance_dict['instance_id']} response not generated")
                    continue
                if len(self.specified_instances) > 0:
                    if instance_dict['instance_id'] not in specified_instances:
                        continue
                    else:
                        specified_instances.remove(instance_dict['instance_id'])      
                
                if self.verbose:
                    print(f"Evaluting instance {instance_dict['instance_id']}")
                llm_response = instance_dict["llm_raw_response"]
                id = instance_dict["instance_id"]
                cur_instance = self.instance.format(id)
                problem = self.get_problem(cur_instance, self.domain_pddl)
                plan_executor = self.get_executor(cur_instance, self.domain_pddl)
                llm_plan = save_gpt3_response(plan_executor, llm_response, self.llm_plan_file)
                instance_dict["extracted_llm_plan"] = llm_plan
                
                correct = int(validate_plan(self.domain_pddl, cur_instance, self.llm_plan_file))
                
                if self.verbose:
                    print(f"Correct: {bool(correct)}")
                instance_dict["llm_correct"] = bool(correct)
                total_correct += correct
                total_instances += 1
                self.save_json(structured_output, task_name)
        if self.verbose:
            print(f"Total correct: {total_correct}")
            print(f"Total instances: {total_instances}")
            print(f"Accuracy: {total_correct/total_instances}")
    
    

    

    


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
    
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    # parser.add_argument('--random_example', type=str, default="False", help='Random example')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    args = parser.parse_args()
    task = args.task
    engine = args.engine
    config = args.config
    verbose = eval(args.verbose)
    ignore_existing = args.ignore_existing
    specified_instances = args.specific_instances

    print(f"Task: {task}, Engine: {engine}, Config: {config}, Verbose: {verbose}")

    # specified_instances = args.specified_instances
    # random_example = eval(args.random_example)
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'
    response_evaluator = ResponseEvaluator(config_file, engine, specified_instances, verbose, ignore_existing)
    eval_plan_dict = {
        't1': 'task_1_plan_generation',
        't1_zero': 'task_1_plan_generation_zero_shot',
        't1_cot': 'task_1_plan_generation_state_tracking',
        }
    eval_plan_pddl_dict = {
        't1_pddl': 'task_1_plan_generation_pddl',
        't1_zero_pddl': 'task_1_plan_generation_zero_shot_pddl',
        }
    if task in eval_plan_dict:
        try:
            task_name = eval_plan_dict[task]
        except:
            raise ValueError("Invalid task name")
        response_evaluator.evaluate_plan(task_name)
    elif task in eval_plan_pddl_dict:
        try:
            task_name = eval_plan_pddl_dict[task]
        except:
            raise ValueError("Invalid task name")
        response_evaluator.evaluate_plan_pddl(task_name)
        
    


            
            
                    

        
        

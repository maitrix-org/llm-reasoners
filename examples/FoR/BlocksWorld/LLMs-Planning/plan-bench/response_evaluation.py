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
                    if 'new_instance' not in instance_dict:
                        correct = int(validate_plan(self.domain_pddl, cur_instance, self.llm_plan_file))
                    else:
                        self.write_new_instance(instance_dict['new_instance'])
                        correct = int(validate_plan('pr-new-domain.pddl', 'pr-new-problem.pddl', self.llm_plan_file))
                        #remove new_instance key from instance_dict
                        del instance_dict['new_instance']
                    if 'optimality' in task_name:
                        if correct:
                            cost = get_cost_gpt_3(llm_response)
                            plan_list = [len(pl) > 0 for pl in llm_plan.split('\n')]
                            actual_cost_llm = sum(plan_list)
                            instance_dict["actual_cost_of_llm_plan"] = actual_cost_llm
                            instance_dict["cost_by_llm"] = cost
                            if actual_cost_llm == plan_executor.cost:
                                correct = True
                            else:
                                correct = False
                except:
                    # Plan extraction failed
                    correct = int(False)
                    print(f"Warning: Plan extraction failed for plan {id}")
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
    

    

    def evaluate_state(self, task_name):
        structured_output = self.load_json(task_name)
        total_correct = 0
        total_instances = 0
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
                ground_state = instance_dict["ground_truth_plan"]
                llm_state = text_to_state(llm_response, self.data)
                
                if sorted(ground_state) == sorted(llm_state):
                    correct = True
                else:
                    correct = False
                instance_dict["extracted_llm_plan"] = llm_state
                instance_dict["llm_correct"] = bool(correct)
                total_correct += correct
                total_instances += 1
                if self.verbose:
                    print(f"Correct: {bool(correct)}")
                self.save_json(structured_output, task_name)
        if self.verbose:
            print(f"Total correct: {total_correct}")
            print(f"Total instances: {total_instances}")
            print(f"Accuracy: {total_correct/total_instances}")
            


    def parse_output(self, action_set, output):
        output_dict = {}
        goal_cond = False
        precond_act = False
        precond_act_flag = False
        precond_pred = False
        for line in output.split('\n'):
            if '[STATEMENT]' in line:
                break
            if line.strip() == "":
                continue
            if goal_cond:
                output_dict['unmet_goal'] = text_to_state(line.strip(), self.data)
                goal_cond = False
                continue
            if precond_act:
                _ , action = text_to_plan(line.strip(), action_set, self.llm_plan_file, self.data)
                output_dict['unmet_precondition']['action'] = action
                precond_act = False
                precond_act_flag = True
                continue
            if precond_act_flag and precond_pred:
                # print(line.strip(), text_to_state(line.strip(), self.data))
                output_dict['unmet_precondition']['predicate'] = text_to_state(line.strip(), self.data)
                precond_pred = False
                precond_act_flag = False

            if 'plan is valid' in line:
                if 'valid' not in output_dict:
                    output_dict['valid'] = True
                break
            elif 'plan is invalid' in line:
                output_dict['valid'] = False
            if 'unmet goal' in line and 'unmet precondition' in line:
                break
            if 'unmet goal' in line:
                output_dict['unmet_goal'] = ''
                goal_cond = True
            elif 'unmet precondition' in line:
                if 'action' in line:
                    output_dict['unmet_precondition'] = {}
                    precond_act = True
                else:
                    precond_pred = True
            elif 'Unmet precondition:' in line:
                if precond_act_flag:
                    output_dict['unmet_precondition']['predicate'] = text_to_state(line.strip(), self.data)
                    precond_act_flag = False

        return output_dict
    
    def evaluate_verification(self, task_name):
        structured_output = self.load_json(task_name)
        total_correct_binary = 0
        total_correct_w_type = 0
        total_correct_w_expl = 0
        total_instances = 0
        for instance_dict in structured_output["instances"]:
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
                id = instance_dict["instance_id"]
                cur_instance = self.instance.format(id)
                problem = self.get_problem(cur_instance, self.domain_pddl)
                # plan_executor = self.get_executor(cur_instance, self.domain_pddl)
                llm_response = instance_dict["llm_raw_response"]
                ground_truth_response = instance_dict["ground_truth_plan"]
                correct_binary = False
                correct_w_type = False
                correct_w_expl = False
                parsed_llm_response = self.parse_output(problem.actions, llm_response)
                parsed_ground_truth_response = self.parse_output(problem.actions, ground_truth_response)
                instance_dict["extracted_llm_plan"] = parsed_llm_response
                instance_dict["parsed_ground_truth_plan"] = parsed_ground_truth_response
                if parsed_llm_response['valid'] == parsed_ground_truth_response['valid']:
                    correct_binary = True
                    if not parsed_llm_response['valid']:
                        # print(sorted(list(parsed_llm_response.keys())), sorted(list(parsed_ground_truth_response.keys())), sorted(list(parsed_llm_response.keys())) == sorted(list(parsed_ground_truth_response.keys())))
                        if sorted(list(parsed_llm_response.keys())) == sorted(list(parsed_ground_truth_response.keys())):
                            correct_w_type = True
                            if 'unmet_goal' in parsed_llm_response:
                                # if parsed_ground_truth_response['unmet_goal'] == parsed_llm_response['unmet_goal']:
                                if any([llm_pred in parsed_ground_truth_response['unmet_goal'] for llm_pred in parsed_llm_response['unmet_goal']]):
                                    correct_w_expl = True
                            if 'unmet_precondition' in parsed_llm_response:
                                try:
                                    if parsed_llm_response['unmet_precondition']['action'] == parsed_ground_truth_response['unmet_precondition']['action']:
                                        if any([llm_pred in parsed_ground_truth_response['unmet_precondition']['predicate'] for llm_pred in parsed_llm_response['unmet_precondition']['predicate']]):
                                            correct_w_expl = True
                                        
                                except KeyError:
                                    print(f"For Instance {id}")
                                    print(parsed_llm_response)
                                    print(parsed_ground_truth_response)
                                    # raise KeyError
                    else:
                        correct_w_type = True
                        correct_w_expl = True
                    
                                
                instance_dict['llm_correct_binary'] = correct_binary
                instance_dict['llm_correct_w_type'] = correct_w_type
                instance_dict['llm_correct_w_expl'] = correct_w_expl
                total_correct_binary += correct_binary
                total_correct_w_type += correct_w_type
                total_correct_w_expl += correct_w_expl
                total_instances += 1
                if self.verbose:
                    print(f"Correct binary: {correct_binary}")
                    print(f"Correct w type: {correct_w_type}")
                    print(f"Correct w expl: {correct_w_expl}")
                self.save_json(structured_output, task_name)
        if self.verbose:
            print(f"Total correct binary: {total_correct_binary}")
            print(f"Total correct w type: {total_correct_w_type}")
            print(f"Total correct w expl: {total_correct_w_expl}")
            print(f"Total instances: {total_instances}")
            print(f"Accuracy binary: {total_correct_binary/total_instances}")
            print(f"Accuracy w type: {total_correct_w_type/total_instances}")
            print(f"Accuracy w expl: {total_correct_w_expl/total_instances}")
        
    


if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to run \
    \n t1 = Plan Generation\
    \n t2 = Optimal Planning \
    \n t3 = Plan Verification \
    \n t4 = Plan Reuse\
    \n t5 = Plan Generalization\
    \n t6 = Replanning (easier) \
    \n t7 = Reasoning about Plan Execution \
    \n t8_1 = Goal Reformulation (Goal shuffling) \
    \n t8_2 = Goal Reformulation (Full -> Partial) \
    \n t8_3 = Goal Reformulation (Partial -> Full) \
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
        't2': 'task_2_plan_optimality',
        
        't4': 'task_4_plan_reuse',
        't5': 'task_5_plan_generalization',
        't6': 'task_6_replanning',
        't8_1': 'task_8_1_goal_shuffling',
        't8_2': 'task_8_2_full_to_partial',
        't8_3': 'task_8_3_partial_to_full',
    }
    eval_state_dict = {
        't7': 'task_7_plan_execution'
    }
    eval_verification_dict = {
        't3': 'task_3_plan_verification',
        't3_1': 'task_3_plan_verification_with_llm_plans'
    }
    if task in eval_plan_dict:
        try:
            task_name = eval_plan_dict[task]
        except:
            raise ValueError("Invalid task name")
        response_evaluator.evaluate_plan(task_name)
    elif task in eval_state_dict:
        try:
            task_name = eval_state_dict[task]
        except:
            raise ValueError("Invalid task name")
        response_evaluator.evaluate_state(task_name)
    elif task in eval_verification_dict:
        try:
            task_name = eval_verification_dict[task]
        except:
            raise ValueError("Invalid task name")
        response_evaluator.evaluate_verification(task_name)
    


            
            
                    

        
        

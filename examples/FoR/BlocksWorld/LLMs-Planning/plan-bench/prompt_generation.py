import os
import random

import yaml
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time
import json

from tqdm import tqdm


class PromptGenerator:
    def __init__(self,config_file, verbose, ignore_existing, seed) -> None:
        self.n_examples = 1
        self.output_dir = "prompts"
        self.verbose = verbose
        self.ignore_existing = ignore_existing
        self.plan_file = "sas_plan"
        self.data = self.read_config(config_file)
        self.instance_dir = self.data['instance_dir']
        self.domain_pddl = f'./instances/{self.data["domain_file"]}'
        self._set_task_params()
        self._set_seed(seed)

    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def _set_task_params(self, instance_dir=None):
        if instance_dir is None:
            instance_dir = self.instance_dir
        else:
            self.instance_dir = instance_dir
        self.instance_folder = f'./instances/{instance_dir}/'
        self.instance = f'./instances/{instance_dir}/{self.data["instances_template"]}'
        self.n_files = min(self.data['n_instances'], len(os.listdir(self.instance_folder)))

        self.i_start = self.data['start']
        self.i_end = self.data['end']
    
    def compute_plan(self, domain, instance):
        fast_downward_path = os.getenv("FAST_DOWNWARD")
        # Remove > /dev/null to see the output of fast-downward
        assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
        cmd = f"{fast_downward_path}/fast-downward.py {domain} {instance} --search \"astar(lmcut())\" > /dev/null 2>&1"
        os.system(cmd)

        if not os.path.exists(self.plan_file):
            return ""
        return Path(self.plan_file).read_text()

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def get_problem(self, instance, domain):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        return reader.parse_instance(instance)

    def get_executor(self, instance, domain, ground=False):
        plan_executor = Executor(domain, instance, ground=ground)
        return plan_executor

    def save_json(self, output_file, structured_output):
        os.makedirs(f"{self.output_dir}/{self.data['domain_name']}/", exist_ok=True)
        with open(f"{self.output_dir}/{self.data['domain_name']}/" + output_file + ".json", "w") as f:
            json.dump(structured_output, f, indent=4)
    
    def load_json(self, output_file, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        if self.ignore_existing:
            return None
        if os.path.exists(f"{output_dir}/{self.data['domain_name']}/" + output_file + ".json"):
            with open(f"{output_dir}/{self.data['domain_name']}/" + output_file + ".json", "r") as f:
                return json.load(f)
        else:
            return None
    def load_results_json(self, output_file):
        output_dir = "results"
        engine = "gpt-4_chat"
        assert os.path.exists(f"{output_dir}/{self.data['domain_name']}/{engine}/" + output_file + ".json"), "File does not exist"
        with open(f"{output_dir}/{self.data['domain_name']}/{engine}/" + output_file + ".json", "r") as f:
            return json.load(f)
        
    
        # ========================================== TASKS ========================================== #
    def task_1_plan_generation(self, specified_instances=[], random_example=False):
        task_name = f"task_1_plan_generation"
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        completed_instances =  []
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        if len(specified_instances):
            range_list = []
            for specified_instance in specified_instances:
                range_list.append(specified_instance - self.n_examples)
        else:
            range_list = range(self.i_start, self.i_end + 2 - self.n_examples)
        
        for start in tqdm(range_list):
            if start + self.n_examples in completed_instances: 
                continue
            query = self.data["domain_intro"]
            instance_structured_output = {}
            examples = []
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan                
                if last_plan:
                    cur_instance = self.instance.format(i)
                    instance_structured_output["instance_id"] = i
                else:
                    if random_example:
                        new_i = random.choice([ln for ln in range(1,self.n_files) if ln != i])
                        cur_instance = self.instance.format(new_i)
                        examples.append(new_i)
                    else:
                        cur_instance = self.instance.format(i)
                        examples.append(i)
                if self.verbose:
                    print(f"Instance {cur_instance}")
                # --------------- Read Instance --------------- #
                problem = self.get_problem(cur_instance, self.domain_pddl)
                # --------------------------------------------- #
                # ------------ Put plan and instance into text ------------ #
                gt_plan = self.compute_plan(self.domain_pddl, cur_instance)
                gt_plan_text = get_plan_as_text(self.data)
                query += fill_template(*instance_to_text(problem, get_plan, self.data))
                # --------------------------------------------------------- #
                
            if self.verbose:
                print(query)

            stop_statement = '[STATEMENT]'
            # Querying LLM
            if 'caesar' in self.data['domain_name']:
                query = caesar_encode(query)
                stop_statement = caesar_encode(stop_statement)
            instance_structured_output["example_instance_ids"] = examples
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = gt_plan_text
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

    def task_2_plan_optimality(self, specified_instances=[], random_example=False):
        task_name = f"task_2_plan_optimality"
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        if len(specified_instances):
            range_list = []
            for specified_instance in specified_instances:
                range_list.append(specified_instance - self.n_examples)
        else:
            range_list = range(self.i_start, self.i_end + 2 - self.n_examples)
        
        for start in tqdm(range_list):
            query = self.data["domain_intro_cost"]
            instance_structured_output = {}
            examples = []
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                
                if last_plan:
                    cur_instance = self.instance.format(i)
                    instance_structured_output["instance_id"] = i
                else:
                    if random_example:
                        new_i = random.choice([ln for ln in range(1,self.n_files) if ln != i])
                        cur_instance = self.instance.format(new_i)
                        examples.append(new_i)
                    else:
                        cur_instance = self.instance.format(i)
                        examples.append(i)
                plan_executor = self.get_executor(cur_instance, self.domain_pddl)
                if self.verbose:
                    print(f"Instance {cur_instance}")
                gt_plan_text = get_plan_as_text(self.data)
                instance_query, plan = optimality(plan_executor, self.data, get_plan)
                query += instance_query
                # --------------------------------------------------------- #
                
            if self.verbose:
                print(query)

            stop_statement = '[STATEMENT]'
            # Querying LLM
            if 'caesar' in self.data['domain_name']:
                query = caesar_encode(query)
                stop_statement = caesar_encode(stop_statement)
            if i in completed_instances:
                continue
            instance_structured_output["example_instance_ids"] = examples
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = gt_plan_text
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

    def task_3_plan_verification(self, specified_instances=[]):
        task_name = "task_3_plan_verification"
        correct_plans = 0
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        if len(specified_instances):
            range_list = specified_instances
        else:
            range_list = range(self.i_start, self.i_end+1)
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])

        for i in tqdm(range_list):
            query = self.data["domain_intro"]
            instance_structured_output = {}
            cur_instance = self.instance.format(i)
            if i in completed_instances:
                    continue
            if self.verbose:
                print(f"Instance {cur_instance}")
            instance_structured_output["instance_id"] = i
            example_instances = random.choices([ln for ln in range(1,self.n_files) if ln != i], k=3)        
            example_type = [-1, 0, 1]
            random.shuffle(example_type)
            for example, example_type in zip(example_instances, example_type):
                example_instance = self.instance.format(example)
                plan_executor = self.get_executor(example_instance, self.domain_pddl)
                text,_ = plan_verification(plan_executor, example_type, self.data, True)
                query += text
            instance_type = random.choice([-1, 0, 1])
            plan_executor = self.get_executor(cur_instance, self.domain_pddl)
            text, answer = plan_verification(plan_executor, instance_type, self.data, False)
            query += text
            if self.verbose:
                print(query)

            stop_statement = '[STATEMENT]'
            
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = answer
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)
    
    def task_3_plan_verification_with_llm_plans(self, specified_instances=[]):
        task_name = "task_3_plan_verification_with_llm_plans"
        llm_plan_task_name = 'task_1_plan_generation'
        correct_plans = 0
        llm_plan_json = self.load_results_json(llm_plan_task_name)
        
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        if len(specified_instances):
            range_list = specified_instances
        else:
            range_list = range(self.i_start, self.i_end+1)
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])

        for i in tqdm(range_list):
            query = self.data["domain_intro"]
            instance_structured_output = {}
            cur_instance = self.instance.format(i)
            if i in completed_instances:
                    continue
            llm_plan = []
            for llm_plan_instance in llm_plan_json["instances"]:
                if llm_plan_instance["instance_id"] == i:
                    llm_plan = llm_plan_instance["extracted_llm_plan"]
                    break
            if len(llm_plan) == 0:
                continue            
            if self.verbose:
                print(f"Instance {cur_instance}")
            instance_structured_output["instance_id"] = i
            example_instances = random.choices([ln for ln in range(1,self.n_files) if ln != i], k=3)        
            example_type = [-1, 0, 1]
            random.shuffle(example_type)
            for example, example_type in zip(example_instances, example_type):
                example_instance = self.instance.format(example)
                plan_executor = self.get_executor(example_instance, self.domain_pddl)
                text,_ = plan_verification(plan_executor, example_type, self.data, True)
                query += text
            instance_type = random.choice([-1, 0, 1])
            plan_executor = self.get_executor(cur_instance, self.domain_pddl)
            text, answer = plan_verification(plan_executor, instance_type, self.data, False, llm_plan=llm_plan)
            query += text
            if self.verbose:
                print(query)

            stop_statement = '[STATEMENT]'
            
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = answer
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

    def task_3_zero_shot_plan_verification(self, specified_instances=[]):
        task_name = "task_3_zero_shot_plan_verification"
        llm_plan_task_name = 'task_1_plan_generation'
        correct_plans = 0
        llm_plan_json = self.load_results_json(llm_plan_task_name)
        
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        if len(specified_instances):
            range_list = specified_instances
        else:
            range_list = range(self.i_start, self.i_end+1)
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])

        for i in tqdm(range_list):
            query = self.data["domain_intro"]
            instance_structured_output = {}
            cur_instance = self.instance.format(i)
            if i in completed_instances:
                    continue
            llm_plan = []
            for llm_plan_instance in llm_plan_json["instances"]:
                if llm_plan_instance["instance_id"] == i:
                    llm_plan = llm_plan_instance["extracted_llm_plan"]
                    break
            if len(llm_plan) == 0:
                continue            
            if self.verbose:
                print(f"Instance {cur_instance}")
            instance_structured_output["instance_id"] = i
            example_instances = random.choices([ln for ln in range(1,self.n_files) if ln != i], k=3)        
            example_type = [-1, 0, 1]
            random.shuffle(example_type)
            for example, example_type in zip(example_instances, example_type):
                example_instance = self.instance.format(example)
                plan_executor = self.get_executor(example_instance, self.domain_pddl)
                text,_ = plan_verification(plan_executor, example_type, self.data, True)
                query += text
            instance_type = random.choice([-1, 0, 1])
            plan_executor = self.get_executor(cur_instance, self.domain_pddl)
            text, answer = plan_verification(plan_executor, instance_type, self.data, False, llm_plan=llm_plan)
            query += text
            if self.verbose:
                print(query)

            stop_statement = '[STATEMENT]'
            
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = answer
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

        
    
    def task_4_plan_reuse(self, specified_instances=[]):
        # n = self.i_end - self.i_start + 1
        task_name = "task_4_plan_reuse"
        correct_plans = 0
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        if len(specified_instances):
            range_list = specified_instances
        else:
            range_list = range(self.i_start, self.i_end+1)
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        for i in tqdm(range_list):
            instance_structured_output = {}
            cur_instance = self.instance.format(i)
            if i in completed_instances:
                    continue
            instance_structured_output["instance_id"] = i
            plan_executor = self.get_executor(cur_instance, self.domain_pddl)
            problem = self.get_problem(cur_instance, self.domain_pddl)
            # print(f"Instance {cur_instance}")
            # gt_plan = self.compute_plan(domain, cur_instance)
            full_plan_query, plan = generate_plan_subset(plan_executor, self.data, True)
            subset_plan_query, plan_subset = generate_plan_subset(plan_executor, self.data, False)
            gt_plan_text = get_plan_as_text(self.data, plan_subset)
            query = self.data["domain_intro"]
            query += full_plan_query
            query += subset_plan_query
            # --------------------------------------------------------- #
            new_model = plan_executor.get_new_instance(change_goal=True, change_init=False)
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = gt_plan_text
            instance_structured_output["new_instance"] = new_model
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)
    
    def task_5_plan_generalization(self, specified_instances=[], random_example=False):
        task_name = f"task_5_plan_generalization"
        self._set_task_params(instance_dir=self.data['generalized_instance_dir'])
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        print(completed_instances)
        if len(specified_instances):
            range_list = []
            for specified_instance in specified_instances:
                range_list.append(specified_instance - self.n_examples)
        else:
            range_list = range(self.i_start, self.i_end + 2 - self.n_examples)
        
        for start in tqdm(range_list):
            query = self.data["domain_intro"]
            instance_structured_output = {}
            examples = []
            is_already_completed = False
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
               
                if last_plan:
                    cur_instance = self.instance.format(i)
                    if i in completed_instances:
                        print("Skipping instance")
                        is_already_completed = True
                        continue
                    instance_structured_output["instance_id"] = i
                else:
                    if random_example:
                        new_i = random.choice([ln for ln in range(1,self.n_files) if ln != i])
                        cur_instance = self.instance.format(new_i)
                        examples.append(new_i)
                    else:
                        cur_instance = self.instance.format(i)
                        examples.append(i)
                if self.verbose:
                    print(f"Instance {cur_instance}")
                # --------------- Read Instance --------------- #
                problem = self.get_problem(cur_instance, self.domain_pddl)
                # --------------------------------------------- #
                # ------------ Put plan and instance into text ------------ #
                gt_plan = self.compute_plan(self.domain_pddl, cur_instance)
                gt_plan_text = get_plan_as_text(self.data)
                query += fill_template(*instance_to_text(problem, get_plan, self.data))
                # --------------------------------------------------------- #
            if is_already_completed:
                continue    
            if self.verbose:
                print(query)
            
            stop_statement = '[STATEMENT]'
            # Querying LLM
            if 'caesar' in self.data['domain_name']:
                query = caesar_encode(query)
                stop_statement = caesar_encode(stop_statement)
            instance_structured_output["example_instance_ids"] = examples
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = gt_plan_text
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

    def task_6_replanning(self, specified_instances=[], random_example=False, harder=0):
        task_name = f"task_6_replanning"
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        if len(specified_instances):
            range_list = []
            for specified_instance in specified_instances:
                range_list.append(specified_instance - self.n_examples)
        else:
            range_list = range(self.i_start, self.i_end + 2 - self.n_examples)
        
        for start in tqdm(range_list):
            query = self.data["domain_intro"]
            instance_structured_output = {}
            examples = []
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                
                if last_plan:
                    cur_instance = self.instance.format(i)
                    instance_structured_output["instance_id"] = i
                else:
                    if random_example:
                        new_i = random.choice([ln for ln in range(1,self.n_files) if ln != i])
                        cur_instance = self.instance.format(new_i)
                        examples.append(new_i)
                    else:
                        cur_instance = self.instance.format(i)
                        examples.append(i)
                plan_executor = self.get_executor(cur_instance, self.domain_pddl)
                if self.verbose:
                    print(f"Instance {cur_instance}")
                
                instance_query, plan, new_model = replanning(plan_executor, self.data, get_plan, harder)
                query += instance_query
                # --------------------------------------------------------- #
                
                    
            if self.verbose:
                print(query)
            gt_plan_text = get_plan_as_text(self.data, plan)
            stop_statement = '[STATEMENT]'
            # Querying LLM
            if 'caesar' in self.data['domain_name']:
                query = caesar_encode(query)
                stop_statement = caesar_encode(stop_statement)
            if i in completed_instances:
                continue
            instance_structured_output["example_instance_ids"] = examples
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = gt_plan_text
            instance_structured_output["new_instance"] = new_model
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

    def task_7_plan_execution(self, specified_instances=[], random_example=False):
        task_name = f"task_7_plan_execution"
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        if len(specified_instances):
            range_list = []
            for specified_instance in specified_instances:
                range_list.append(specified_instance - self.n_examples)
        else:
            range_list = range(self.i_start, self.i_end + 2 - self.n_examples)
        
        for start in tqdm(range_list):
            if start + self.n_examples in completed_instances:
                continue
            query = self.data["domain_intro"]
            instance_structured_output = {}
            examples = []
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                
                if last_plan:
                    cur_instance = self.instance.format(i)
                    instance_structured_output["instance_id"] = i
                else:
                    
                    if random_example:
                        new_i = random.choice([ln for ln in range(1,self.n_files) if ln != i])
                        cur_instance = self.instance.format(new_i)
                        examples.append(new_i)
                    else:
                        cur_instance = self.instance.format(i)
                        examples.append(i)
                plan_executor = self.get_executor(cur_instance, self.domain_pddl)
                if self.verbose:
                    print(f"Instance {cur_instance}")
                if(len(plan_executor.plan) < 1):
                     print("Skipping instance "+str(i)+" becauce it requires an empty plan.")
                     continue
                instance_query, answer = plan_execution(plan_executor, self.data, get_plan)
                query += instance_query
                # --------------------------------------------------------- #
                
                    
            if self.verbose:
                print(query)
            stop_statement = '[STATEMENT]'
            # Querying LLM
            if 'caesar' in self.data['domain_name']:
                query = caesar_encode(query)
                stop_statement = caesar_encode(stop_statement)
            instance_structured_output["example_instance_ids"] = examples
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = answer
            structured_output["instances"].append(instance_structured_output)        
            self.save_json(task_name, structured_output)

    def task_8_1_goal_shuffling(self, specified_instances=[]):
        task_name = f"task_8_1_goal_shuffling"
        n = self.i_end - self.i_start + 1
        skipped = 0
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        if len(specified_instances):
            range_list = specified_instances
        else:
            range_list = range(self.i_start, self.i_end + 1)
        for i in tqdm(range_list):
            cur_instance = self.instance.format(i)
            instance_structured_output = {}
            plan_executor = self.get_executor(cur_instance, self.domain_pddl)
            if i in completed_instances:
                continue
            problem = self.get_problem(cur_instance, self.domain_pddl)
            gt_plan = self.compute_plan(self.domain_pddl, cur_instance)

            if gt_plan == "":
                std_out = f"[-]: Timeout or error gathering Ground Truth plan for {cur_instance}. Continuing..."                
                print(std_out)
                skipped += 1
                continue
            gt_plan_text = get_plan_as_text(self.data)

            number_of_preds, goal_full = paraphrase_goal(plan_executor, self.data)

            try:
                init_specific, goal_specific, plan_specific,_ = instance_to_text(problem, True, self.data)
                init_specific_shuffled, goal_specific_shuffled, _,_ = instance_to_text(problem, True,
                                                                                                 self.data,
                                                                                                 shuffle=True)
            except Exception as e:
                print(f"[-]: Error exception: {e}")
                print(f"[-]: Error converting {cur_instance} to text. Continuing...")
                skipped += 1
                continue
            single_goal_instances = 1 if number_of_preds == 1 else 0
            # =============== Random =============== #
            query = self.data["domain_intro"]
            query += fill_template(init_specific, goal_specific, plan_specific, self.data)
            query += fill_template(init_specific, goal_specific_shuffled, "", self.data)
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = plan_specific
            instance_structured_output["instance_id"] = i
            instance_structured_output["single_goal_instances"] = single_goal_instances
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)
         
    def task_8_2_full_to_partial(self, specified_instances=[]):
        task_name = f"task_8_2_full_to_partial"
        n = self.i_end - self.i_start + 1
        skipped = 0
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        if len(specified_instances):
            range_list = specified_instances
        else:
            range_list = range(self.i_start, self.i_end + 1)
        for i in tqdm(range_list):
            cur_instance = self.instance.format(i)
            if i in completed_instances:
                continue
            plan_executor = self.get_executor(cur_instance, self.domain_pddl)
            instance_structured_output = {}
            problem = self.get_problem(cur_instance, self.domain_pddl)
            gt_plan = self.compute_plan(self.domain_pddl, cur_instance)

            if gt_plan == "":
                std_out = f"[-]: Timeout or error gathering Ground Truth plan for {cur_instance}. Continuing..."                
                print(std_out)
                skipped += 1
                continue
            gt_plan_text = get_plan_as_text(self.data)

            number_of_preds, goal_full = paraphrase_goal(plan_executor, self.data)

            try:
                init_specific, goal_specific, plan_specific, _ = instance_to_text(problem, True, self.data)
                
            except:
                
                print(f"[-]: Error converting {cur_instance} to text. Continuing...")
                skipped += 1
                continue
            single_goal_instances = 1 if number_of_preds == 1 else 0
            # =============== Random =============== #
            query = self.data["domain_intro"]
            query += fill_template(init_specific, goal_full, plan_specific, self.data)
            query += fill_template(init_specific, goal_specific, "", self.data)
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = plan_specific
            instance_structured_output["instance_id"] = i
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

    def task_8_3_partial_to_full(self, specified_instances=[]):
        task_name = "task_8_3_partial_to_full"
        n = self.i_end - self.i_start + 1
        skipped = 0
        instance_structured_outputs = []
        structured_output = self.load_json(task_name)
        
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "prompt_type": "oneshot",
                                "domain": self.data['domain_name'],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        for inst in structured_output["instances"]:
            if inst["query"]:
                completed_instances.append(inst["instance_id"])
        if len(specified_instances):
            range_list = specified_instances
        else:
            range_list = range(self.i_start, self.i_end + 1)
        for i in tqdm(range_list):
            cur_instance = self.instance.format(i)
            if i in completed_instances:
                continue
            plan_executor = self.get_executor(cur_instance, self.domain_pddl)
            instance_structured_output = {}
            problem = self.get_problem(cur_instance, self.domain_pddl)
            gt_plan = self.compute_plan(self.domain_pddl, cur_instance)

            if gt_plan == "":
                std_out = f"[-]: Timeout or error gathering Ground Truth plan for {cur_instance}. Continuing..."                
                print(std_out)
                skipped += 1
                continue
            gt_plan_text = get_plan_as_text(self.data)

            number_of_preds, goal_full = paraphrase_goal(plan_executor, self.data)

            try:
                init_specific, goal_specific, plan_specific, _ = instance_to_text(problem, True, self.data)
                
            except:
                
                print(f"[-]: Error converting {cur_instance} to text. Continuing...")
                skipped += 1
                continue
            single_goal_instances = 1 if number_of_preds == 1 else 0
            # =============== Random =============== #
            query = self.data["domain_intro"]
            query += fill_template(init_specific, goal_specific, plan_specific, self.data)
            query += fill_template(init_specific, goal_full, "", self.data)
            instance_structured_output["query"] = query
            instance_structured_output["ground_truth_plan"] = plan_specific
            instance_structured_output["instance_id"] = i
            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)

if __name__=="__main__":
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Task to run \
    \n t1 = Plan Generation\
    \n t2 = Optimal Planning \
    \n t3 = Plan Verification \
    \n t4 = Plan Reuse\
    \n t5 = Plan Generalization\
    \n t6 = Replanning \
    \n t7 = Reasoning about Plan Execution \
    \n t8_1 = Goal Reformulation (Goal shuffling) \
    \n t8_2 = Goal Reformulation (Full -> Partial) \
    \n t8_3 = Goal Reformulation (Partial -> Full) \
    ')
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    #config
    parser.add_argument('--config', type=str, required=True, help='Config file name (no need to add .yaml)')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--random_example', type=str, default="False", help='Random example')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    task = args.task
    config = args.config
    verbose = eval(args.verbose)
    specified_instances = args.specific_instances
    random_example = eval(args.random_example)
    ignore_existing = args.ignore_existing
    seed = args.seed
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'
    prompt_generator = PromptGenerator(config_file, verbose, ignore_existing, seed)
    if task == 't1':
        prompt_generator.task_1_plan_generation(specified_instances, random_example)
    elif task == 't2':
        prompt_generator.task_2_plan_optimality(specified_instances, random_example)
    elif task == 't3':
        prompt_generator.task_3_plan_verification(specified_instances)
    elif task == 't3_1':
        prompt_generator.task_3_plan_verification_with_llm_plans(specified_instances)
    elif task == 't4':
        prompt_generator.task_4_plan_reuse(specified_instances)
    elif task == 't5':
        prompt_generator.task_5_plan_generalization(specified_instances, random_example)
    elif task == 't6':
        prompt_generator.task_6_replanning(specified_instances, random_example)
    elif task == 't7':
        prompt_generator.task_7_plan_execution(specified_instances, random_example)
    elif task == 't8_1':
        prompt_generator.task_8_1_goal_shuffling(specified_instances)
    elif task == 't8_2':
        prompt_generator.task_8_2_full_to_partial(specified_instances)
    elif task == 't8_3':
        prompt_generator.task_8_3_partial_to_full(specified_instances)



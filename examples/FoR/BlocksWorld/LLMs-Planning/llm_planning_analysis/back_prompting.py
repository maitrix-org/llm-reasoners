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
class BackPrompter():
    """
    Tasks:
    T1. Goal-directed reasoning
    T2. Paraphrasing of goals
    T3. Plan subset completion
    T4. Plan generalization
    T5. Optimality
    T6. Replanning
    T7. Plan execution
    """

    def __init__(self, engine, verbose, ignore_existing):
        self.engine = engine
        self.verbose = verbose
        self.n_examples = 1
        self.ignore_existing = ignore_existing
        self.max_gpt_response_length = 500

        self.plan_file = "sas_plan"
        self.gpt3_plan_file = "gpt_sas_plan"
        if self.engine == 'bloom':
            self.model = self.get_bloom()
        elif 'finetunedgpt3' in self.engine:
            print(self.engine)
            assert self.engine.split(':')[1] is not None
            model = ':'.join(self.engine.split(':')[1:])
            print(model)
            self.engine='finetunedgpt3'
            self.model = {'model':model}
        else:
            self.model = None
        
        

    # ========================================== UTILS ========================================== #
    def compute_plan(self, domain, instance, timeout=30):
        fast_downward_path = os.getenv("FAST_DOWNWARD")
        # Remove > /dev/null to see the output of fast-downward
        assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
        cmd = f"timeout {timeout}s {fast_downward_path}/fast-downward.py {domain} {instance} --search \"astar(lmcut())\" > /dev/null 2>&1"
        os.system(cmd)

        if not os.path.exists(self.plan_file):
            return ""
        return Path(self.plan_file).read_text()

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def get_problem(self, instance, domain):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        return reader.parse_instance(instance)

    def get_executor(self, instance, domain, ground=True):
        plan_executor = Executor(domain, instance, ground=ground)
        return plan_executor

    def get_bloom(self):
        max_memory_mapping = {0: "0GB", 1: "43GB", 2: "43GB", 3: "43GB", 4: "43GB", 5: "43GB"}
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", cache_dir='/data/karthik/LLM_models/bloom/',
                                                     local_files_only=False, load_in_8bit=True, device_map='auto',
                                                     max_memory=max_memory_mapping)
        return {'model': model, 'tokenizer': tokenizer}


    def save_json(self, output_file, structured_output):
        os.makedirs(f"results/{self.data['domain_name']}/{self.engine}/json/", exist_ok=True)
        with open(f"results/{self.data['domain_name']}/{self.engine}/json/" + output_file + ".json", "w") as f:
            json.dump(structured_output, f, indent=4)
    
    def load_json(self, output_file):
        if os.path.exists(f"results/{self.data['domain_name']}/{self.engine}/json/" + output_file + ".json") and not self.ignore_existing:
            with open(f"results/{self.data['domain_name']}/{self.engine}/json/" + output_file + ".json", "r") as f:
                return json.load(f)
        else:
            return None

    

    def task_1_plan_generation_backprompting(self, config_file, use_llm_feedback, specified_instances=[], random_example=False):
        
        
        self.read_config(config_file)
        if use_llm_feedback['use_llm']:
            if use_llm_feedback['zero_shot']:
                task_name = "task_1_plan_generation_backprompting_llm_feedback_zero_shot"
                if use_llm_feedback['val_form']:
                    task_name = "task_1_plan_generation_backprompting_llm_feedback_zero_shot_val_form"
            else:
                task_name = "task_1_plan_generation_backprompting_llm_feedback"
        else:
            task_name = "task_1_plan_generation_backprompting"

        instance_dir = self.data['instance_dir']
        domain_pddl = f'./instances/{self.data["domain_file"]}'
        instance_folder = f'./instances/{instance_dir}/'
        instance = f'./instances/{instance_dir}/{self.data["instances_template"]}'
        n_files = min(self.data['n_instances'], len(os.listdir(instance_folder)))

        i_start = self.data['start']
        i_end = self.data['end']
        n_files = i_end - i_start + 1  # min(self.data['n_instances'], len(os.listdir(instance_folder)))
        final_output = ""
        instance_structured_outputs = []

        failed_instances = 0
        structured_output = self.load_json(task_name)
        if structured_output is None:
            structured_output = {
                                "task": task_name,
                                "engine": self.engine,
                                "prompt_type": "oneshot",
                                "domain": self.data["domain_name"],
                                "instances": instance_structured_outputs,
                                }
        completed_instances =  []
        prev_success_results = {}
        for inst in structured_output["instances"]:
            if not inst["could_not_extract"]:
                completed_instances.append(inst["instance_id"])
                prev_success_results[inst["instance_id"]] = inst["act_correct"]
            else:
                if [msg["content"] for msg in inst["messages"] if msg["role"] == "assistant"][-1]:
                    completed_instances.append(inst["instance_id"])
                    prev_success_results[inst["instance_id"]] = inst["act_correct"]
        if len(specified_instances):
            range_list = []
            for specified_instance in specified_instances:
                range_list.append(specified_instance - self.n_examples)
        else:
            range_list = range(i_start, i_end + 2 - self.n_examples)
        
        for ind, start in enumerate(range_list):
            query = self.data["domain_intro"]
            instance_structured_output = {}
            examples = []
            for i in range(start, start + self.n_examples+1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                if last_plan:
                    cur_instance = instance.format(i)
                else:
                    if random_example:
                        new_i = random.choice([ln for ln in range(1,self.n_files) if ln != i])
                        cur_instance = instance.format(new_i)
                        examples.append(new_i)
                    else:
                        cur_instance = instance.format(i)
                
                # --------------- Add to final output --------------- #
                final_output += f"\n Instance {cur_instance}\n"
                if self.verbose:
                    print(f"Instance {cur_instance}")
                # if 'logistics' in self.data['domain_name']:
                #     plan_executor = self.get_executor(cur_instance, domain_pddl, ground=False)
                # else:
                #     plan_executor = self.get_executor(cur_instance, domain_pddl)
                # # --------------- Read Instance --------------- #
                problem = self.get_problem(cur_instance, domain_pddl)
                # # --------------------------------------------- #
                # # ------------ Put plan and instance into text ------------ #
                gt_plan = self.compute_plan(domain_pddl, cur_instance)
                plan = get_plan_as_text(self.data)
                # query += fill_template(*instance_to_text(problem, get_plan, self.data))
                # instance_text, _ = generate_plan_cot(plan_executor, self.data, get_plan)
                # # query_inst, _ = generate_plan_subset_cot(plan_executor, self.data, False)
                # query += instance_text
                # query += query_inst
                query += fill_template(*instance_to_text(problem, get_plan, self.data))
                
                if get_plan:
                    examples.append(i)
                else:
                    # Store instance id for the actual instance LLM is asked to solve
                    instance_structured_output["instance_id"] = i
                    
                # stop_statement = '[STATEMENT]'
            
            instance_structured_output["example_instance_ids"] = examples
            # print(query)
            # llm_response = ""
            if instance_structured_output["instance_id"] in completed_instances:
                print(f"Instance {instance_structured_output['instance_id']} already completed")
                if not prev_success_results[instance_structured_output["instance_id"]]:
                    failed_instances += 1
                continue
            # if self.is_already_correct(instance_structured_output["instance_id"]):
            #     print(f"Instance {instance_structured_output['instance_id']} already completed and correct")
            #     continue
            
            messages, llm_plan, verifier_states_correct, act_correct, steps, context_window_hit, could_not_extract, feedback_messages = \
                self.get_repeated_verification(self.engine, query, domain_pddl, problem, cur_instance, use_llm_feedback)
            instance_structured_output["messages"] = messages
            instance_structured_output["steps"] = steps
            instance_structured_output["verifier_states_correct"] = bool(verifier_states_correct)
            instance_structured_output["act_correct"] = bool(act_correct)
            instance_structured_output["extracted_llm_plan"] = llm_plan
            instance_structured_output["context_window_hit"] = bool(context_window_hit)
            instance_structured_output["could_not_extract"] = bool(could_not_extract)
            instance_structured_output["feedback_messages"] = feedback_messages

            if not bool(act_correct):
                failed_instances += 1

            structured_output["instances"].append(instance_structured_output)
            self.save_json(task_name, structured_output)
            
        try:
            os.remove(self.plan_file)
            os.remove(self.gpt3_plan_file)
        except:
            print("No plan file to delete")

        # --------------- Add to final output --------------- #

        
        structured_output["failed_instances"] = failed_instances
        self.save_json(task_name, structured_output)
        final_output += f"[+]: The number of correct plans is {n_files - failed_instances}/{n_files}={(n_files - failed_instances) / (n_files) * 100}%"
        print(final_output)

        return structured_output

    def is_already_correct(self, instance_id):
        try:
            old_json = self.load_json("task1_reasoning")
            for instance in old_json["instances"]:
                if instance["instance_id"] == instance_id:
                    return instance["correct"]        
        except Exception as e:
            print(f"Error: {e}")
            return False
        
    def get_llm_feedback(self, domain_pddl, llm_plan, cur_instance, use_llm_feedback, messages=[]):
        '''
        Has an LLM correct the plan. Previous messages with the LLM (generation messages)
        can optionally be passed as context to the LLM if there's interest
        in seeing this as a conversation.
        '''
        query = self.data["domain_intro"]
        instance_dir = self.data['instance_dir']
        instance_format = f'./instances/{instance_dir}/{self.data["instances_template"]}'
        instance_folder = f'./instances/{instance_dir}/'
        n_files = min(self.data['n_instances'], len(os.listdir(instance_folder)))
        instance_structured_output = {}
        instance_id = int(cur_instance.split('/')[-1].split('.')[0].split('-')[-1])
        instance_structured_output["instance_id"] = instance_id
        if not use_llm_feedback['zero_shot']:
            example_instances = random.choices([ln for ln in range(1, n_files) if ln != instance_id], k=3)        
            example_type = [-1, 0, 1]
            random.shuffle(example_type)
            for example, example_type in zip(example_instances, example_type):
                example_instance = instance_format.format(example)
                plan_executor = self.get_executor(example_instance, domain_pddl)
                text,_ = plan_verification(plan_executor, self.data, True, give_response = True, example_type=example_type)
                query += text
            plan_executor = self.get_executor(cur_instance, domain_pddl)
            text, _ = plan_verification(plan_executor, self.data, False, llm_plan=llm_plan)
        else:
            plan_executor = self.get_executor(cur_instance, domain_pddl)
            if not use_llm_feedback['val_form']:
                text = plan_verification_zero_shot(plan_executor, self.data, llm_plan=llm_plan)
            else:
                text = plan_verification_zero_shot_val_form(plan_executor, self.data, llm_plan=llm_plan)
        query += text

        response, messages, _, _ = send_query_with_feedback(query, engine, messages, system_message="You are the planner assistant that validates whether a provided plan is correct and provides feedback if not.")

        return response, messages


    #TODO: self-critique
    def self_critique(self, engine, original_query, domain_pddl, problem, cur_instance, use_llm_feedback, threshold_feedback_amount=15):
        correct = 0
        steps = 0
        query = original_query
        # print(original_query)
        messages = []
        could_not_extract = False
        current_flag = 0
        print(f"Sending query to LLM: Instance {cur_instance}")
        while correct==0 and steps < threshold_feedback_amount:      

            pass


            llm_response, messages, context_window_hit, rate_limit_hit = send_query_with_feedback(query, engine, messages)
#             llm_response = """
# unstack the red block from on top of the blue block
# put down the red block
# unstack the blue block from on top of the yellow block
# put down the blue block
# unstack the yellow block from on top of the orange block
# pick up the red block
# stack the red block on top of the yellow block
# [PLAN END]
#             """
#             context_window_hit, rate_limit_hit = False, False
            if context_window_hit:
                break
            if rate_limit_hit:
                print("Rate limit hit. Waiting for 60 seconds.")
                time.sleep(60)
                continue

            feedback_messages = []

            if use_llm_feedback['use_llm']:
                try:
                    llm_plan, readable_plan = text_to_plan(llm_response, problem.actions, self.gpt3_plan_file, self.data, ground_flag=True)
                except:
                    could_not_extract = True 
                    break
                query, feedback_messages = self.get_llm_feedback(domain_pddl, llm_plan.strip().split("\n"), cur_instance, use_llm_feedback)
                verifier_states_correct = False
                for line in query.split('\n'):
                    if 'plan is valid' in line.lower():
                        correct = True
                        break
            else:
                try: 
                    _, readable_plan = text_to_plan(llm_response, problem.actions, self.gpt3_plan_file, self.data)
                    feedback_dict = get_val_feedback(domain_pddl, cur_instance, self.gpt3_plan_file)
                except:
                    could_not_extract = True 
                    break
                verifier_states_correct = int(feedback_dict["validation_info"]["is_valid_plan"])
                query = get_validation_message(feedback_dict, self.data)
            steps += 1

        # Determine whether backprompting result is actually correct
        # Since the LLM is not sound, this may not be true when the LLM says it is correct
        # This is always true when VAL says it is correct
        if use_llm_feedback['use_llm']:
            try: 
                _, readable_plan = text_to_plan(llm_response, problem.actions, self.gpt3_plan_file, self.data)
                feedback_dict = get_val_feedback(domain_pddl, cur_instance, self.gpt3_plan_file)
            except:
                print("WARNING: final plan could not be verified")
            act_correct = int(feedback_dict["validation_info"]["is_valid_plan"])
        else:
            act_correct = verifier_states_correct

        # print(f"Final LLM response after {steps} steps")
        return messages, readable_plan, verifier_states_correct, act_correct, steps, context_window_hit, could_not_extract, feedback_messages
    



    def get_repeated_verification(self, engine, original_query, domain_pddl, problem, cur_instance, use_llm_feedback, threshold_feedback_amount=15):
        correct = 0
        steps = 0
        query = original_query
        # print(original_query)
        messages = []
        could_not_extract = False
        print(f"Sending query to LLM: Instance {cur_instance}")
        while correct==0 and steps < threshold_feedback_amount:            
            llm_response, messages, context_window_hit, rate_limit_hit = send_query_with_feedback(query, engine, messages)
#             llm_response = """
# unstack the red block from on top of the blue block
# put down the red block
# unstack the blue block from on top of the yellow block
# put down the blue block
# unstack the yellow block from on top of the orange block
# pick up the red block
# stack the red block on top of the yellow block
# [PLAN END]
#             """
#             context_window_hit, rate_limit_hit = False, False
            if context_window_hit:
                break
            if rate_limit_hit:
                print("Rate limit hit. Waiting for 60 seconds.")
                time.sleep(60)
                continue

            feedback_messages = []

            if use_llm_feedback['use_llm']:
                try:
                    llm_plan, readable_plan = text_to_plan(llm_response, problem.actions, self.gpt3_plan_file, self.data, ground_flag=True)
                except:
                    could_not_extract = True 
                    break
                query, feedback_messages = self.get_llm_feedback(domain_pddl, llm_plan.strip().split("\n"), cur_instance, use_llm_feedback)
                verifier_states_correct = False
                for line in query.split('\n'):
                    if 'plan is valid' in line.lower():
                        correct = True
                        break
            else:
                try: 
                    _, readable_plan = text_to_plan(llm_response, problem.actions, self.gpt3_plan_file, self.data)
                    feedback_dict = get_val_feedback(domain_pddl, cur_instance, self.gpt3_plan_file)
                except:
                    could_not_extract = True 
                    break
                verifier_states_correct = int(feedback_dict["validation_info"]["is_valid_plan"])
                query = get_validation_message(feedback_dict, self.data)
            steps += 1

        # Determine whether backprompting result is actually correct
        # Since the LLM is not sound, this may not be true when the LLM says it is correct
        # This is always true when VAL says it is correct
        if use_llm_feedback['use_llm']:
            try: 
                _, readable_plan = text_to_plan(llm_response, problem.actions, self.gpt3_plan_file, self.data)
                feedback_dict = get_val_feedback(domain_pddl, cur_instance, self.gpt3_plan_file)
            except:
                print("WARNING: final plan could not be verified")
            act_correct = int(feedback_dict["validation_info"]["is_valid_plan"])
        else:
            act_correct = verifier_states_correct

        # print(f"Final LLM response after {steps} steps")
        return messages, readable_plan, verifier_states_correct, act_correct, steps, context_window_hit, could_not_extract, feedback_messages
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    #config
    parser.add_argument('--config', type=str, required=True, help='Config file name (no need to add .yaml)')
    parser.add_argument('--engine', type=str, required=True, help='Engine to use \
                        \n gpt-4_chat = GPT-4 \
                        \n bloom = Bloom \
                        \n gpt-3.5-turbo_chat = GPT-3.5 Turbo \
                        \n davinci = GPT-3 Davinci \
                        \n curie = GPT-3 Curie \
                        \n babbage = GPT-3 Babbage \
                        \n ada = GPT-3 Ada \
                        ')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--random_example', type=str, default="False", help='Random example')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--llm_validation', action='store_true', help='Use LLM to validate itself instead of VAL')
    
    args = parser.parse_args()
    config = args.config
    engine = args.engine
    verbose = eval(args.verbose)
    specified_instances = args.specific_instances
    random_example = eval(args.random_example)
    ignore_existing = args.ignore_existing
    seed = args.seed
    random.seed(seed)
    # print(task, config, verbose, specified_instances, random_example)
    use_llm_feedback = {
        'use_llm': args.llm_validation,
        'zero_shot': True,
        'val_form': True
    }
    config_file = f'./configs/{config}.yaml'
    backprompter = BackPrompter(engine, verbose=verbose, ignore_existing=ignore_existing)
    backprompter.task_1_plan_generation_backprompting(config_file, use_llm_feedback, specified_instances=specified_instances, random_example=random_example)



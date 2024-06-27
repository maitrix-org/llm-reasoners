import random
import argparse
import os
from prompt_generation import PromptGenerator
from response_evaluation import ResponseEvaluator
from response_generation import ResponseGenerator

"""
TODO: 
1. logistics text_to_state
2. obfuscations deceptive and random
3. logistics plan generalization instances ========= DONE
4. problem generators with filters
"""
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
    
    parser.add_argument('--run_till_completion', type=str, default="False", help='Run till completion')
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    parser.add_argument('--ignore_existing', action='store_true', help='Ignore existing output')
    parser.add_argument('--specific_instances', nargs='+', type=int, default=[], help='List of instances to run')
    parser.add_argument('--random_example', type=str, default="False", help='Random example')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    task = args.task
    config = args.config
    engine = args.engine
    verbose = eval(args.verbose)
    specified_instances = args.specific_instances
    seed=args.seed
    ignore_existing = args.ignore_existing
    random_example = eval(args.random_example)
    run_till_completion = eval(args.run_till_completion)
    # print(task, config, verbose, specified_instances, random_example)
    config_file = f'./configs/{config}.yaml'


    # ========================= Prompt Generation =========================
    assert os.path.exists(config_file), f"Config file {config_file} does not exist"
    prompt_generator = PromptGenerator(config_file, verbose, ignore_existing, seed)
    if task == 't1':
        prompt_generator.task_1_plan_generation(specified_instances, random_example)
    elif task == 't1_zero':
        prompt_generator.task_1_plan_generation_zero_shot(specified_instances, random_example)
    elif task == 't1_cot':
        prompt_generator.task_1_plan_generation_state_tracking(specified_instances, random_example)
    elif task == 't1_pddl':
        prompt_generator.task_1_plan_generation_pddl(specified_instances, random_example)
    elif task == 't1_zero_pddl':
        prompt_generator.task_1_plan_generation_zero_shot_pddl(specified_instances, random_example)
    else:
        raise NotImplementedError
    
    # ========================= Response Generation =========================
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


    # ========================= Response Evaluation =========================
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



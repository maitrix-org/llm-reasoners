import random
import argparse
import os
from prompt_generation import PromptGenerator
from response_evaluation import ResponseEvaluator
from response_generation import ResponseGenerator


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
    prompt_generator = PromptGenerator(config_file, verbose, ignore_existing, seed)
    if task == 't1':
        prompt_generator.task_1_plan_generation(specified_instances, random_example)
    elif task == 't2':
        prompt_generator.task_2_plan_optimality(specified_instances, random_example)
    elif task == 't3':
        prompt_generator.task_3_plan_verification(specified_instances)
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
    
    # ========================= Response Generation =========================
    response_generator = ResponseGenerator(config_file, engine, verbose, ignore_existing)
    task_dict = {
        't1': 'task_1_plan_generation',
        't2': 'task_2_plan_optimality',
        't3': 'task_3_plan_verification',
        't4': 'task_4_plan_reuse',
        't5': 'task_5_plan_generalization',
        't6': 'task_6_replanning',
        't7': 'task_7_plan_execution',
        't8_1': 'task_8_1_goal_shuffling',
        't8_2': 'task_8_2_full_to_partial',
        't8_3': 'task_8_3_partial_to_full',
    }
    try:
        task_name = task_dict[task]
    except:
        raise ValueError("Invalid task name")
    response_generator.get_responses(task_name, run_till_completion=run_till_completion)

    # ========================= Response Evaluation =========================
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
        't3': 'task_3_plan_verification'
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



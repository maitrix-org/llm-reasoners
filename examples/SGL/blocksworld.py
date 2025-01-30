import os
import shutil
from multiprocessing import Pool
import reasoners.benchmark.bw_utils as bw_utils
from reasoners import Reasoner
from reasoners.benchmark import BWEvaluator
import json
import time
import argparse
from examples.RAP.blocksworld.world_model import BlocksWorldModel
from examples.RAP.blocksworld.search_config import BWConfig
from reasoners.algorithm import MCTS
from reasoners.lm import SGLangModel
from reasoners.benchmark.blocksworld import rap_bw_extractor

os.environ["SGLANG_API_URL"] = "http://127.0.0.1:30001"
model_name = "meta-llama/Llama-3.1-8B"
model = SGLangModel(model_name)

with open('examples/CoT/blocksworld/prompts/pool_prompt_v1.json') as f:
    prompt = json.load(f)

evaluator = BWEvaluator(config_file='examples/CoT/blocksworld/data/bw_config.yaml',
                        domain_file='examples/CoT/blocksworld/data/generated_domain.pddl',
                        data_path='examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json',
                        init_prompt=prompt)

prompt = evaluator.sample_prompt(shuffle_prompt=False, num_shot=4)

cot_inputs = []

for example in evaluator.full_dataset:
    cot_inputs.append(prompt['icl'].replace('<init_state>', example["init"])
                            .replace('<goals>', example["goal"])
                            .replace('<action>', ''))

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory) 
    os.makedirs(directory)

def parse_args():
    parser = argparse.ArgumentParser(description='Run BlocksWorld with CoT or RAP approach')
    parser.add_argument('--method', type=str, choices=['cot', 'rap'], required=True,
                       help='Method to use: cot (Chain of Thought) or rap (Reasoning and Planning)')
    return parser.parse_args()

def process_input_cot(args):
    index, input_data = args
    
    output = model.generate([input_data],
                             hide_input=True,
                             eos_token_id=['\n[']).text[0][:-1].strip()
    
    output_dir = "CoT_results"
    output_file = os.path.join(output_dir, f"output_{index}.txt")
    
    with open(output_file, 'w') as file:
        file.write(output)
    
    return output_file

world_model = BlocksWorldModel(base_model=model, prompt=prompt, max_steps=4)
config = BWConfig(base_model=model, prompt=prompt)
algorithm = MCTS(depth_limit=4, disable_tqdm=False, output_trace_in_each_iter=True, n_iters=10)
reasoner_rap = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)

def process_example_RAP(args):
    index, example_input = args

    # Reasoning process for each example
    result = reasoner_rap(example_input)

    # Save the result to an output file
    output_dir = "RAP_results"
    output_file = os.path.join(output_dir, f"output_{index}.txt")

    with open(output_file, 'w') as file:
        file.write(str(rap_bw_extractor(result)))

    return output_file

if __name__ == "__main__":
    args = parse_args()
    
    # start time
    start = time.time()
    
    num_processes = 100
    output_dir = "CoT_results" if args.method == 'cot' else "RAP_results"
    clear_directory(output_dir)

    if args.method == 'cot':
        inputs_with_indices = list(enumerate(cot_inputs))
        process_func = process_input_cot
    else:  # rap
        inputs_with_indices = list(enumerate(evaluator.full_dataset))
        process_func = process_example_RAP

    with Pool(num_processes) as pool:
        results = pool.map(process_func, inputs_with_indices)
    
    acc = 0
    time_taken = time.time() - start
    
    print("Evaluating the results")
    
    for i in range(len(cot_inputs)):
        output = open(f"{output_dir}/output_{i}.txt").read()
        acc += evaluator.eval_output(evaluator.full_dataset[i], output)
    
    print(f"Time taken for reasoning: {time_taken}")
    print(f"Accuracy: {acc / len(cot_inputs)}")
    
    # Cleanup
    for i in range(len(cot_inputs)):
        os.remove(results[i])
        
    os.rmdir(output_dir)
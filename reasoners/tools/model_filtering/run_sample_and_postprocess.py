import argparse
import os
import datasets
import json
import transformers
import numpy as np
from verl.utils.data_process.utils import save_dataset
from verl.utils.data_process.filter import LengthFilter


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

def filter_dataset_by_length(dataset, max_length) -> datasets.Dataset:
    """
    Filter the dataset by the prompt length.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    prompt_length_filter = LengthFilter(tokenizer=tokenizer, max_length=args.max_prompt_tokens)
    print(f"Before filtering by prompt length {max_length}: num_items = {len(dataset)}")
    dataset = dataset.filter(lambda x: prompt_length_filter.check(x), num_proc=64)
    print(f"After filtering by prompt length {max_length}: num_items = {len(dataset)}")
    return dataset


def postprocess_dataset(dataset, dataset_name) -> datasets.Dataset:
    """
    Postprocess (patch) the dataset for the given domain.
    """
    # Normalize the data source for 'logic' domain
    if "zebra_puzzle" in dataset_name or "ordering_puzzle" in dataset_name or "graph" in dataset_name:
        dataset = dataset.map(lambda item: {
            "data_source": "logic__" + item["data_source"]
        })
    
    # Refine the prompt for 'table' domain
    if "table" in dataset_name:
        original_instruction = "Please output the final answer within \\boxed{}. If there are multiple answers, please output them separated by |."
        refined_instruction = "Please output the final answer in \\boxed{}. If there are multiple answers, include them in a single box, separated by |."
        dataset = dataset.map(lambda item: {
            "prompt": [
                {"role": "user", "content": item["prompt"][0]["content"].replace(original_instruction, refined_instruction)},
            ]
        })
        
    # Add 'question' key for 'stem' domain as it's required in the reward
    if "stem" in dataset_name:
        # add 'question' to 'extra_info'
        dataset = dataset.map(lambda item: {
            "extra_info": {
                **item["extra_info"],
                "question": item['raw_prompt']
            }
    })
    
    # Keep 8 unit tests, otherwise the reward model will be too slow
    # Truncate (right) too long input prompts, usually with very long unit test io pairs. 
    if "codegen" in dataset_name:
        def _map_func(item):
            
            MAX_PROMPT_CHARS = 4096 * 5 # Use char count instead of token count as it's faster; 4096 is max input tokens and 5 is avg. chars per token
            if len(item['prompt'][0]['content']) > MAX_PROMPT_CHARS:
                print(f"Truncating {item['id']} from {dataset_name} to {MAX_PROMPT_CHARS} chars")
                item['prompt'][0]['content'] = item['prompt'][0]['content'][:MAX_PROMPT_CHARS]
            
            MAX_UNIT_TEST_CHARS = 1024 * 128  # max unit test input/output length is 128KB
            ground_truth = json.loads(item['reward_model']['ground_truth'])
            if 'inputs' in ground_truth and len(ground_truth['inputs']) > 0:
                # Store complete inputs/outputs in ground truth
                complete_inputs = ground_truth['inputs']
                complete_outputs = ground_truth['outputs']
                
                complete_inputs_after_filtered_too_long = []  
                complete_outputs_after_filtered_too_long = []
                for ut_in, ut_out in zip(complete_inputs, complete_outputs):
                    if len(ut_in) > MAX_UNIT_TEST_CHARS:
                        pass
                        print(f"Unit test input is too long, {len(ut_in)} chars")
                    elif len(ut_out) > MAX_UNIT_TEST_CHARS:
                        pass
                        print(f"Unit test output is too long, {len(ut_out)} chars")
                    else:
                        complete_inputs_after_filtered_too_long.append(ut_in)
                        complete_outputs_after_filtered_too_long.append(ut_out)

                # Get 8 random indices from this test case's inputs/outputs
                n_tests = len(complete_inputs_after_filtered_too_long)
                sample_indices = np.random.choice(n_tests, min(8, n_tests), replace=False)

                # Update ground truth with sampled data while preserving complete data
                ground_truth['complete_inputs'] = complete_inputs_after_filtered_too_long
                ground_truth['complete_outputs'] = complete_outputs_after_filtered_too_long
                ground_truth['inputs'] = [complete_inputs_after_filtered_too_long[i] for i in sample_indices]
                ground_truth['outputs'] = [complete_outputs_after_filtered_too_long[i] for i in sample_indices]

                item['reward_model']['ground_truth'] = json.dumps(ground_truth)
                
                if n_tests > 8:
                    print(f"Filtering {n_tests} unit tests to 8 for {item['id']} from {dataset_name}")
                    print(type(item['reward_model']['ground_truth']))
                    print(item['reward_model']['ground_truth'])
                
            return item
        
        dataset = dataset.map(_map_func, num_proc=16)
    return dataset


def sample_dataset(dataset, target_sample_size) -> datasets.Dataset:
    """
    Sample the dataset to the target sample size.
    """
    if target_sample_size != -1:
        subset_sample_size = int(target_sample_size * subset_sample_sizes[input_data_name] / total_sample_size)

        sampled_dataset = dataset.shuffle(seed=GLOBAL_SEED).select(range(subset_sample_size))
        print(f"{len(sampled_dataset)} items sampled from {input_data_name}")
    else:
        sampled_dataset = dataset
        subset_sample_size = len(dataset)
    return sampled_dataset, subset_sample_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str, default="/mnt/weka/home/zhuojun.cheng/leo/Reasoning360/data/train_filtered")
    parser.add_argument("--input_data_names", type=str, nargs="+")
    parser.add_argument("--output_data_dir", type=str, default="/mnt/weka/home/zhuojun.cheng/leo/Reasoning360/data/train_guru15k")
    parser.add_argument("--target_sample_size", type=int, required=True)
    parser.add_argument("--domain", type=str, default="math", choices=["math", "codegen", "simulation", "logic", "table", "stem"])
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    args = parser.parse_args()

    # Load data
    print(args.input_data_names)
    total_sample_size = 0
    subset_sample_sizes = {}
    dataset_list = []
    for input_data_name in args.input_data_names:
        data_path = os.path.join(args.input_data_dir, f"{input_data_name}.parquet")
        print(f"Loading {input_data_name} from {data_path}")
        try:
            dataset = datasets.Dataset.from_parquet(data_path)
        except Exception as e:
            # for nested columns, e.g., livecodebench
            import polars as pl
            dataframe = pl.read_parquet(data_path).to_pandas()
            dataset = datasets.Dataset.from_pandas(dataframe)
        print(f"Loaded {input_data_name} from {data_path}, with {len(dataset)} items")
        subset_sample_sizes[input_data_name] = len(dataset)
        total_sample_size += len(dataset)
        dataset_list.append(dataset)
        
    # Postprocessing and sample
    os.makedirs(args.output_data_dir, exist_ok=True)
    for input_data_name, dataset in zip(args.input_data_names, dataset_list):
        
        # Filter by length
        if args.max_prompt_tokens is not None:
            dataset = filter_dataset_by_length(dataset, args.max_prompt_tokens)
        # Misc postprocess (patch some issues not covered in previous data processing)
        dataset = postprocess_dataset(dataset, input_data_name)
        # Sample to `target_sample_size`
        sampled_dataset, subset_sample_size = sample_dataset(dataset, args.target_sample_size)
        
        # Print the first 5 items
        for idx, item in enumerate(sampled_dataset):
            if idx < 5:
                print(f"========== {input_data_name} item#{idx} ==========")
                try:
                    print(json.dumps(item, indent=4))
                except:
                    print(item)
            else:
                break
        
        # Save the sampled dataset
        output_path = save_dataset(
            sampled_dataset, 
            args.output_data_dir, 
            f"{args.domain}__{input_data_name.split('__')[1]}_sampled", 
            subset_sample_size
            )
        print(f"Saving {output_path} with {subset_sample_size} items")

    print(f"Saved {args.output_data_dir} with {args.target_sample_size} items")

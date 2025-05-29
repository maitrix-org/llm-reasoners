import os
import datasets
import argparse
from sklearn.model_selection import train_test_split

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset

# Instruction for zebra puzzle answers
InstructionFollow = "Output the grid in the form of a dictionary with keys as header containing a list of the attributes and rows denoting each row of the final grid. Please return the final answer in <answer> </answer> tags, for example <answer> {\"header\": [\"Position\", \"Nationality\", \"Job\"], \"rows\": [[\"1\", \"british\", \"plumber\"], [\"2\", \"polish\", \"carpenter\"]]} </answer>."

def make_prefix(dp):
    clues = dp['clues']
    if isinstance(clues, list):
        clues = "\n".join(clues)
    result = dp['ground_truth']
    instruction = dp['instruction']
    prefix = f"{instruction} The clues are: {clues}."
    return prefix


def extract_from_ground_truth(text):
    if isinstance(text, dict):
        return text
    else:
        return eval(text)
        
def make_map_fn(split, data_source):
    def process_fn(example, idx):
        question = make_prefix(example)
        grid_size = str(example['config']["cols"]) + "x" + str(example['config']["rows"])
        final_grid = extract_from_ground_truth(example['ground_truth'])
        
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question + " " + InstructionFollow
            }],
            "ability": "logical_reasoning", 
            "reward_model": {
                    "style": "rule",
                    "ground_truth": final_grid,
                },
            "apply_chat_template": True,
            "extra_info": {
                'id': example['id'] if 'id' in example else str(idx),
                "grid_size": grid_size,
                'raw_instruction': example['instruction'],
                'raw_input': example['clues'],
                'split': split,
            }
        }
        
        if idx == 0:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data
        
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='data/raw/zebra_puzzles.json', help='Path to json file')
    parser.add_argument('--output_dir', default='data', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set')
    parser.add_argument('--data_source', default='zebra_puzzle_dataset', help='Name of data source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load dataset from JSON
    dataset = datasets.load_dataset('json', data_files=args.json_path)['train']
   
    # Transform dataset
    process_train_fn = make_map_fn('train', args.data_source)
    processed_dataset = dataset.map(function=process_train_fn, with_indices=True)
    
    if args.train_size + args.test_size > 1.0:
        raise ValueError(f"The sum of train_size ({args.train_size}) and test_size ({args.test_size}) cannot exceed 1.0")

    # Split dataset into train and test
    train_indices, test_indices = train_test_split(
        range(len(dataset)), 
        train_size=args.train_size,
        test_size=args.test_size, 
        random_state=args.seed
    )
  
    # Create train and test datasets
    train_dataset = processed_dataset.select(train_indices)
    test_dataset = processed_dataset.select(test_indices)
    
    # Store the original training dataset size
    original_train_size = len(train_dataset)
    
    # Sample the training dataset if needed
    train_dataset = sample_dataset(train_dataset, args.train_sample_size)
    
    # Create output directories
    train_output_dir = os.path.join(args.output_dir, "train")
    test_output_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Save train dataset
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=f"logic__{args.data_source}",
        sample_size=args.train_sample_size if args.train_sample_size else len(train_dataset)
    )
    
    # Save test dataset
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=f"logic__{args.data_source}",
        sample_size=len(test_dataset)
    )
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=args.output_dir, dst=args.hdfs_dir)
            print(f"Data copied to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("HDFS utilities not available. Install verl package for HDFS support.")
            
    print(f"Done! \n"
          f"Train data saved to {train_output_path}\n"
          f"Test data saved to {test_output_path}")
    print(f"Original train set size: {original_train_size} examples")
    print(f"Final train set size: {len(train_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")
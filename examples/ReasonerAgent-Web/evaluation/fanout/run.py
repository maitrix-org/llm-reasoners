import argparse
from glob import glob
import os
import pandas as pd

from evaluator import FanOutQAEvaluator

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('job_name', type=str)
    parser.add_argument('--browsing_data_dir', type=str, 
                        default=os.path.join(current_dir, '..', '..', 'browsing_data'))
    parser.add_argument('--groundtruth_path', type=str, 
                        default=os.path.join(current_dir, '..', '..', 'data', 'fanout-final-dev.json'))
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=9999)
    
    args = parser.parse_args()
    
    evaluator = FanOutQAEvaluator(args.groundtruth_path, start_idx=args.start_idx, end_idx=args.end_idx)

    my_evaluator_log_paths = glob(os.path.join(args.browsing_data_dir, args.job_name + '*.json'))

    scores, records = evaluator.evaluate_batch(my_evaluator_log_paths)
    print(scores)
    
    records_df = pd.DataFrame(records)
    # raw_acc_scores_df = pd.DataFrame(raw_acc_scores)
    # records_df = records_df.merge(raw_acc_scores_df, on='id')
    print(records_df.outcome.value_counts())
    
    output_dir = os.path.join(current_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_name = f'{args.job_name}_{args.start_idx}_{args.end_idx}_eval.csv'
    records_df.to_csv(os.path.join(output_dir, output_name), index=False)

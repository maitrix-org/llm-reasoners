import argparse
from glob import glob
import os
import pandas as pd
import json

from evaluator import FlightSearchEvaluator

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('job_name', type=str)
    parser.add_argument('--browsing_data_dir', type=str, 
                        default=os.path.join(current_dir, '..', '..', 'browsing_data'))
    parser.add_argument('--questions_path', type=str, 
                        default=os.path.join(current_dir, '..', '..', 'task_data', 'flightqa_counterfactual.csv'))
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=9999)
    
    args = parser.parse_args()
    
    evaluator = FlightSearchEvaluator(args.questions_path, args.start_idx, args.end_idx)

    browsing_data_paths = glob(os.path.join(args.browsing_data_dir, args.job_name + '*.json'))
    
    records = []
    for browsing_data_path in browsing_data_paths:
        basename = os.path.basename(browsing_data_path)
        print(basename)
        eval_record = evaluator.evaluate(browsing_data_path)
        if eval_record:
            records.append(eval_record)
        # {'goal': goal, 
        # 'constraints': constraints, 
        # 'observations': observations,
        # 'message': message, 
        # 'outcome': outcome, 
        # 'grounded': grounded, 
        # 'relevant': relevant,
        # 'llm_output': ans_dict}
    records_df = pd.DataFrame(records)
    records_df['correct'] = records_df['grounded'] & records_df['relevant']
    
    print(records_df[['correct', 'grounded', 'relevant']].mean())
    print(records_df.outcome.value_counts())

    output_dir = os.path.join(current_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_name = f'{args.job_name}_{args.start_idx}_{args.end_idx}_eval.json'
    json.dump(records, open(os.path.join(output_dir, output_name), 'w'))
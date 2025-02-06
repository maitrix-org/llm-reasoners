import argparse
import collections
import json
import os

import scipy.stats as stats

parser = argparse.ArgumentParser(description='Calculate average reward.')
parser.add_argument(
    'output_pre', type=str, help='path to pre dictionary containing output.jsonl'
)
parser.add_argument(
    'output_post', type=str, help='path to post dictionary containing output.jsonl'
)

args = parser.parse_args()

if __name__ == '__main__':
    out_path_pre = os.path.join(args.output_pre, 'output.jsonl')
    env_ids_pre = []
    results_pre = []
    with open(out_path_pre, 'r') as f:
        for line in f:
            data = json.loads(line)
            env_ids_pre.append(int((data['instance_id']).split('.')[-1]))
            results_pre.append(data['test_result'])
    results_pre = [x for _, x in sorted(zip(env_ids_pre, results_pre))]

    out_path_post = os.path.join(args.output_post, 'output.jsonl')
    env_ids_post = []
    results_post = []
    with open(out_path_post, 'r') as f:
        for line in f:
            data = json.loads(line)
            env_ids_post.append(int((data['instance_id']).split('.')[-1]))
            results_post.append(data['test_result'])
    results_post = [x for _, x in sorted(zip(env_ids_post, results_post))]

    assert collections.Counter(env_ids_pre) == collections.Counter(env_ids_post)
    total_num = len(env_ids_pre)
    print('Total number of tasks: ', total_num)

    print('Success Rate Pre: ', sum(results_pre) / total_num)
    print('Success Rate Post: ', sum(results_post) / total_num)

    # print(results_pre)
    # print(results_post)
    ttest_results = stats.ttest_rel(results_pre, results_post)
    print(
        f'Paired Sample-T Test: statistic {round(ttest_results.statistic, 4)}; p-value {round(ttest_results.pvalue, 4)}'
    )

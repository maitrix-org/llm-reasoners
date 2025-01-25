import argparse
import json
import os

import browsergym.webarena  # noqa F401 register webarena tasks as gym environments

parser = argparse.ArgumentParser(description='Calculate average reward.')
# parser.add_argument('output_path', type=str, help='path to output.jsonl')
parser.add_argument(
    'output_root', type=str, help='path to dictionary containing output.jsonl'
)
parser.add_argument('--index_success', '-v', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    # env_ids = [
    #     id for id in gym.envs.registry.keys() if id.startswith('browsergym/webarena')
    # ]
    total_reward = 0
    total_num = 0
    out_path = os.path.join(args.output_root, 'output.jsonl')
    success_cases = []
    with open(out_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            total_num += 1
            total_reward += data['test_result']
            if args.index_success and data['test_result'] > 0:
                success_cases.append(int(data['instance_id'].split('.')[-1]))
    if args.index_success:
        print(f'Successful instances: {str(sorted(success_cases))}')

    avg_reward = total_reward / total_num
    print('Success Rate: ', avg_reward)

    print('Number of tasks finished: ', total_num)
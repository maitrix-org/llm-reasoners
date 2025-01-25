import argparse
import base64
from datetime import datetime
from glob import glob
import io
import json
import os
import signal
import sys

from reasoners import ReasonerAgent
from baseline import BrowsingAgent
from utils.llm import LLM
from utils.browser import get_serializable_obs, TimeoutException, timeout_handler
from utils.datasets import get_dataset
from utils.logger import get_agent_logger

import gymnasium as gym

DEBUG = int(os.environ.get('DEBUG', 0))

__SLOW_MO = None
__HEADLESS = True
__TIMEOUT = 5000
__VIEWPORT = {'width': 1280, 'height': 720}
__WAIT_FOR_USER_MESSAGE = False

model_info = {
    'gpt-4o': ('https://api.openai.com/v1/', 'openai'),
    'Meta-Llama-3.1-70B-Instruct': ('http://localhost:8000/v1', 'openai')
}

agent_dict = {
    'reasoner': ReasonerAgent,
    'openhands': BrowsingAgent
}


def main(job_name, 
         model, 
         api_key, 
         output_dir,
         agent,
         config_name,
         max_steps,
         timeout,
         goal=None,
         gym_env_name=None):
    base_url, custom_llm_provider = model_info[model]
    llm = LLM(model=model,
              api_key=api_key,
              base_url=base_url,
              custom_llm_provider=custom_llm_provider)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    log_filename = f'{timestamp}.log'
    logger = get_agent_logger(log_filename)
    agent_cls = agent_dict[agent]
    agent = agent_cls(llm, config_name=config_name, logger=logger)

    os.makedirs(output_dir, exist_ok=True)
    if DEBUG > 0:
        browsergym_instance_dir = os.path.join(output_dir, job_name.split('/')[-1])
        os.makedirs(browsergym_instance_dir, exist_ok=True)
        os.environ["DEBUG_LOG_FOLDER"] = browsergym_instance_dir
    
    if goal is not None:
        env = gym.make(
            'browsergym/openended',
            task_kwargs={'start_url': 'about:blank', 'goal': goal},
            wait_for_user_message=__WAIT_FOR_USER_MESSAGE,
            headless=__HEADLESS,
            slow_mo=__SLOW_MO,
            viewport=__VIEWPORT,
            timeout=__TIMEOUT,
            # disable_env_checker=True,
        )
        env = env.env.env
        obs, info = env.reset()
    else:
        env = gym.make(gym_env_name)
        env = env.env.env
        obs, info = env.reset()
        goal = obs['goal']

    print('Environment started')
    history = []
    error = ''
    action = ''
    step_count = 0
    rewards = []

    while not action.startswith('send_msg_to_user') and step_count < max_steps:
        serializable_obs = get_serializable_obs(env, obs)
        action, thoughts = agent.step(serializable_obs)
        
        history.append((serializable_obs, action, thoughts))
        
        signal.signal(signal.SIGALRM, timeout_handler)
        # Start the alarm
        signal.alarm(timeout)
        
        try:
            # Wait for the result within the specified timeout
            obs, reward, terminated, truncated, info = env.step(action)
            if agent.config['eval_mode']:
                rewards.append(reward)
        except TimeoutException:
            print(f"Environment step timed out after {timeout} seconds")
            error = f"Environment step timed out after {timeout} seconds"
            break
        except Exception as e:
            print('Error when trying to take an action: %s', e)
            error = str(e)
            break
        finally:
            # Disable the alarm after the function call
            signal.alarm(0)
        
        step_count += 1
        
    is_complete = (action.startswith('send_msg_to_user') \
                   and action not in ["send_msg_to_user('Error encountered when browsing.')",
                                      "send_msg_to_user('Too many errors encountered. Task failed.')"])
    
    session_data = {
        'goal': goal,
        'instance_id': gym_env_name,
        'history': history,
        'is_complete': is_complete,
        'error': error,
    }
    if agent.config['eval_mode']:
        if rewards == []:
            rewards = [0.0]
        session_data['rewards'] = rewards
        session_data['test_result'] = float(max(rewards) > 0)
        with open(os.path.join(output_dir, 'output.jsonl'), 'a') as f:
            f.write(json.dumps({
                'instance_id': gym_env_name,
                'goal': goal,
                'test_result': session_data['test_result']
            }) + '\n')
        output_dir = os.path.join(output_dir, "visualize_logs")
        os.makedirs(output_dir, exist_ok=True)

    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_filename = job_name + '_' + current_datetime + '.json'
    with open(os.path.join(output_dir, output_filename), 'w') as f:
        json.dump(session_data, f)


if __name__ == '__main__':
    default_api_key_path = os.path.join(os.path.dirname(__file__), 'default_api_key.txt')
    default_api_key = None
    if os.path.exists(default_api_key_path):
        with open(default_api_key_path, 'r') as fr:
            default_api_key = fr.read().strip()

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run inference on your model with a given dataset."
    )
    
    # Job arguments
    parser.add_argument('job_name', type=str)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=30)

    # IO arguments
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=9999999)
    parser.add_argument('--output_dir', type=str, default='./browsing_data')
    
    # WebArena sampling arguments
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # LLM arguments
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--api_key', type=str, default=default_api_key)
    
    # Agent arguments
    parser.add_argument('--agent', type=str, default='reasoner')
    parser.add_argument('--config_name', type=str, default='browsergym')
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.dataset != 'webarena':
        questions = get_dataset(args.dataset, args.data_root)
        
        for i in range(args.start_idx, min(args.end_idx, len(questions))):
            instruction = questions[i]
            job_name = args.job_name + f'_{i}'
            if glob(os.path.join(args.output_dir, f'{job_name}_*.json')) == []:
                main(job_name, 
                    args.model,
                    args.api_key,
                    args.output_dir,
                    args.agent,
                    args.config_name,
                    args.max_steps,
                    args.timeout,
                    goal=instruction)
            else:
                print(f"Existing log detected for {job_name}, skipping ...")
    else:
        import browsergym.webarena
        env_ids = [
            id for id in gym.envs.registry.keys() if id.startswith('browsergym/webarena')
        ]
        if args.shuffle:
            import random
            random.Random(args.seed).shuffle(env_ids)
        env_ids = sorted(env_ids[args.start_idx:args.end_idx], key=lambda s: int(s.split('.')[-1]))
        for env_id in env_ids:
            job_name = env_id.split('/')[-1]
            if glob(os.path.join(args.output_dir, "visualize_logs", f'{job_name}_*.json')) == []:
                main(job_name, 
                    args.model,
                    args.api_key,
                    args.output_dir,
                    args.agent,
                    args.config_name,
                    args.max_steps,
                    args.timeout,
                    gym_env_name=env_id)
            else:
                print(f"Existing log detected for {job_name}, skipping ...")
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

from litellm.exceptions import BadRequestError

import gymnasium as gym

DEBUG = int(os.environ.get('DEBUG', 0))

__SLOW_MO = None
__HEADLESS = True
__TIMEOUT = 5000
__VIEWPORT = {'width': 1280, 'height': 720}
__WAIT_FOR_USER_MESSAGE = False

model_info = {
    'gpt-4o': ('https://api.openai.com/v1/', 'openai'),
    'o1': ('https://api.openai.com/v1/', 'openai'),
    'o3-mini': ('https://api.openai.com/v1/', 'openai'),
    "deepseek-chat": ("https://api.deepseek.com", "deepseek"),
    'deepseek-reasoner': ("https://api.deepseek.com", "deepseek")
}

agent_dict = {
    'reasoner': ReasonerAgent,
    'openhands': BrowsingAgent
}

retry_if_exception_type = (BadRequestError)

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
    if model in model_info:
        base_url, custom_llm_provider = model_info[model]
        llm = LLM(model=model,
                api_key=api_key,
                base_url=base_url,
                custom_llm_provider=custom_llm_provider)
    elif os.path.isfile(model):
        with open(model) as f:
            model_config = json.load(f)
        llm = {}
        for module, model_name in model_config.items():
            base_url, custom_llm_provider = model_info[model_name]
            llm[module] = LLM(model=model_name,
                api_key=api_key,
                base_url=base_url,
                custom_llm_provider=custom_llm_provider)
    else:
        raise RuntimeError(f"Model {model} is neither supported nor a config file")
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
            if agent.config.get('eval_mode', False):
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
    if agent.config.get('eval_mode', False):
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

    current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_filename = job_name + '_' + current_datetime + '.json'
    with open(os.path.join(output_dir, output_filename), 'w') as f:
        json.dump(session_data, f)


if __name__ == '__main__':
    default_api_key_path = os.path.join(os.path.dirname(__file__), 'default_api_key.txt')
    default_api_key = None
    if os.path.exists(default_api_key_path):
        with open(default_api_key_path, 'r') as fr:
            key = fr.read().strip()
            if key != '':
                default_api_key = key

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run inference on your model with a given dataset."
    )
    
    # Job arguments
    parser.add_argument('job_name', type=str)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--max_retry', type=int, default=0)

    # IO arguments
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
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
    assert args.dataset is not None or args.query is not None, "Please provide a dataset or a query."
    assert args.api_key is not None, "Please provide an API key by either passing it as an argument or saving it in default_api_key.txt."

    main_args = {
        'model': args.model,
        'api_key': args.api_key,
        'output_dir': args.output_dir,
        'agent': args.agent,
        'config_name': args.config_name,
        'max_steps': args.max_steps,
        'timeout': args.timeout,
    }

    if args.query is not None:
        main(
            **{"job_name": args.job_name, 'goal': args.query},
            **main_args
        )
    else:
        goal_key = 'gym_env_name' if args.dataset == 'webarena' else 'goal'
        questions = get_dataset(
            args.dataset, 
            args.data_root, 
            args.shuffle, 
            args.seed, 
            args.start_idx,
            args.end_idx
        )
        for i, question in enumerate(questions):
            idx = i + args.start_idx
            job_name = args.job_name + f'_{idx}'
            if args.dataset == 'webarena':
                job_name = env_id.split('/')[-1]

            if glob(os.path.join(args.output_dir, f'{job_name}_*.json')) == []:
                for attempt in range(args.max_retry+1):
                    try:
                        main(
                            **{"job_name": job_name, goal_key: question},
                            **main_args
                        )
                    except retry_if_exception_type as e:
                        print(f"Error encountered: {str(e)}")
                        print(f"Task failed for {attempt} times, retrying ...")
                    else:
                        break
                else:
                    raise RuntimeError("Max attempts reached, keep getting exceptions.")
            else:
                print(f"Existing log detected for {job_name}, skipping ...")

import argparse
import base64
from datetime import datetime
from glob import glob
import io
import json
import os
import signal
import sys

from utils.llm import LLM
from web_agent import WebAgent
from baseline.openhands_browsing_agent import BrowsingAgent
from web_datasets import get_dataset

import gymnasium as gym
# import browsergym.webarena 
import numpy as np
from PIL import Image

__SLOW_MO = None
__HEADLESS = True
__TIMEOUT = 5000
__VIEWPORT = {'width': 1280, 'height': 720}
__WAIT_FOR_USER_MESSAGE = False

model2baseurl = {
    'gpt-4o': 'https://api.openai.com/v1/',
    'Meta-Llama-3.1-70B-Instruct': 'http://localhost:8000/v1'
}

agent_dict = {
    'ours': WebAgent,
    'openhands': BrowsingAgent
}

def get_scroll_position(page):
    return page.evaluate("""() => {
        const scrollTop = window.scrollY;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const remainingPixels = documentHeight - (scrollTop + windowHeight);

        return {
            'scrollTop': scrollTop,
            'windowHeight': windowHeight,
            'documentHeight': documentHeight,
            'remainingPixels': remainingPixels
        };
    }""")
    
def image_to_jpg_base64_url(
    image: np.ndarray | Image.Image, add_data_prefix: bool = False
):
    """Convert a numpy array to a base64 encoded jpeg image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
    width, height = image.size
    # logger.info(f'Width: {width}, Height: {height}')
    buffered = io.BytesIO()
    image.save(buffered, format='JPEG', quality=10)

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return (
        f'data:image/jpeg;base64,{image_base64}'
        if add_data_prefix
        else f'{image_base64}'
    )
    
def get_serializable_obs(env, obs):
    scroll_position = get_scroll_position(env.page)
    obs['scroll_position'] = scroll_position
    # make observation serializable
    obs['screenshot'] = image_to_jpg_base64_url(obs['screenshot'])
    obs['active_page_index'] = obs['active_page_index'].item()
    obs['elapsed_time'] = obs['elapsed_time'].item()
    return obs
    
def main(job_name, 
         model, 
         api_key, 
         output_dir,
         agent='ours',
         goal=None,
         gym_env_name=None,
         gym_env_config=None,
         use_llama=False,
         use_world_model_planning=False,
        #  use_intent_only_memory=False,
        #  use_no_memory_encoder=False,
        #  use_prompted_memory=False,
        #  use_no_memory_actor=False,
         use_state_memory_encoder=False,
         memory_type='step_key_value',
         encoder_prompt_type='with_memory',
         policy_prompt_type='no_update',
         actor_prompt_type='with_memory',
         world_model_prompt_type='no_update',
         planner_search_num_actions=5,
         planner_search_depth=1,
         planner_critic_num_samples=20,):

    if goal is None and (gym_env_name is None or gym_env_config is None):
        raise RuntimeError("Must provide either goal or gym_env_name.")
    
    llm = LLM(model=model,
              api_key=api_key,
              base_url=model2baseurl.get(model),
              custom_llm_provider='openai',)
    agent_cls = agent_dict[agent]
    agent = agent_cls(llm, 
                  use_llama=use_llama,
                  use_world_model_planning=use_world_model_planning,
                  #   use_intent_only_memory=use_intent_only_memory,
                  #   use_no_memory_encoder=use_no_memory_encoder,
                  #   use_prompted_memory=use_prompted_memory,
                  #   use_no_memory_actor=use_no_memory_actor,
                  use_state_memory_encoder=use_state_memory_encoder,
                  memory_type=memory_type,
                  encoder_prompt_type=encoder_prompt_type,
                  policy_prompt_type=policy_prompt_type,
                  actor_prompt_type=actor_prompt_type,
                  world_model_prompt_type=world_model_prompt_type,
                  planner_search_num_actions=planner_search_num_actions,
                  planner_search_depth=planner_search_depth,
                  planner_critic_num_samples=planner_critic_num_samples,)
    
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
    else:
        env = gym.make(f"browsergym/{gym_env_name}")
    print('Environment started')
    env = env.env.env
    history = []
    error = ''
    obs, info = env.reset()
    action = ''
    step_count = 0
    while not action.startswith('send_msg_to_user') and step_count < 30:
        # print(dir(env.env.env))
        # print(env.env.env)
        serializable_obs = get_serializable_obs(env, obs)
        action, thoughts = agent.step(serializable_obs)
        
        history.append((serializable_obs, action, thoughts))
        
        # Define a custom exception for timeout
        class TimeoutException(Exception):
            pass

        # Function to handle the alarm signal
        def timeout_handler(signum, frame):
            raise TimeoutException("Environment step timed out")

        timeout = 30
        signal.signal(signal.SIGALRM, timeout_handler)
        # Start the alarm
        signal.alarm(timeout)
        
        try:
            # Wait for the result within the specified timeout
            obs, reward, terminated, truncated, info = env.step(action)
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
        
        # print(json.dumps(thoughts, indent=2))
        # print(action)
        step_count += 1
        
    is_complete = (action.startswith('send_msg_to_user') \
                   and action not in ["send_msg_to_user('Error encountered when browsing.')",
                                      "send_msg_to_user('Too many errors encountered. Task failed.')"])
    
    if goal is None:
        session_data = {
            'gym_env_config': gym_env_config,
            'history': history,
            'is_complete': is_complete,
            'error': error,
        }
    else:
        session_data = {
            'goal': goal,
            'history': history,
            'is_complete': is_complete,
            'error': error,
        }
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

    # Add arguments
    parser.add_argument('job_name', type=str)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./task_data/')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--agent', type=str, default='ours')
    parser.add_argument('--api_key', type=str, default=default_api_key)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=9999999)
    parser.add_argument('--output_dir', type=str, default='./browsing_data')
    
    parser.add_argument('--use_llama', action='store_true')
    parser.add_argument('--use_world_model_planning', action='store_true')
    # parser.add_argument('--use_intent_only_memory', action='store_true')
    # parser.add_argument('--use_no_memory_encoder', action='store_true')
    # parser.add_argument('--use_prompted_memory', action='store_true')
    # parser.add_argument('--use_no_memory_actor', action='store_true')
    parser.add_argument('--use_state_memory_encoder', action='store_true')
    parser.add_argument('--memory_type', type=str, default='step_key_value')
    parser.add_argument('--encoder_prompt_type', type=str, default='with_memory')
    parser.add_argument('--policy_prompt_type', type=str, default='no_update')
    parser.add_argument('--actor_prompt_type', type=str, default='with_memory')
    parser.add_argument('--world_model_prompt_type', type=str, default='no_update')
    parser.add_argument('--planner_search_num_actions', type=int, default=5)
    parser.add_argument('--planner_search_depth', type=int, default=1)
    parser.add_argument('--planner_critic_num_samples', type=int, default=20)
    
    # Parse the arguments
    args = parser.parse_args()
    
    questions = get_dataset(args.dataset, args.data_root)
        
    for i in range(args.start_idx, min(args.end_idx, len(questions))):
        instruction = questions[i]
        job_name = args.job_name + f'_{i}'
        if glob(os.path.join(args.output_dir, f'{job_name}_*.json')) == []:
            if args.dataset == 'webarena':
                main(job_name, args.model, args.api_key, args.output_dir,
                    gym_env_name=f"webarena.{i}",
                    gym_env_config=instruction,
                    agent=args.agent,
                    use_llama=args.use_llama,
                    use_world_model_planning=args.use_world_model_planning,
                    # use_intent_only_memory=args.use_intent_only_memory,
                    # use_no_memory_encoder=args.use_no_memory_encoder,
                    # use_prompted_memory=args.use_prompted_memory,
                    # use_no_memory_actor=args.use_no_memory_actor,
                    use_state_memory_encoder=args.use_state_memory_encoder,
                    memory_type=args.memory_type,
                    encoder_prompt_type=args.encoder_prompt_type,
                    policy_prompt_type=args.policy_prompt_type,
                    actor_prompt_type=args.actor_prompt_type,
                    planner_search_num_actions=args.planner_search_num_actions,
                    planner_search_depth=args.planner_search_depth,
                    planner_critic_num_samples=args.planner_critic_num_samples)
            else:
                main(job_name, args.model, args.api_key, args.output_dir,
                    agent=args.agent,
                    goal=instruction,
                    use_llama=args.use_llama,
                    use_world_model_planning=args.use_world_model_planning,
                    # use_intent_only_memory=args.use_intent_only_memory,
                    # use_no_memory_encoder=args.use_no_memory_encoder,
                    # use_prompted_memory=args.use_prompted_memory,
                    # use_no_memory_actor=args.use_no_memory_actor,
                    use_state_memory_encoder=args.use_state_memory_encoder,
                    memory_type=args.memory_type,
                    encoder_prompt_type=args.encoder_prompt_type,
                    policy_prompt_type=args.policy_prompt_type,
                    actor_prompt_type=args.actor_prompt_type,
                    planner_search_num_actions=args.planner_search_num_actions,
                    planner_search_depth=args.planner_search_depth,
                    planner_critic_num_samples=args.planner_critic_num_samples)
        else:
            print(f"Existing log detected for {job_name}, skipping ...")
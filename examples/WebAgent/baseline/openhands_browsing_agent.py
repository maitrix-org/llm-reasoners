import os
from datetime import datetime

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm import LLM
from logging import Logger
from .openhands_response_parser import BrowsingResponseParser

USE_NAV = (
    os.environ.get('USE_NAV', 'true') == 'true'
)  # only disable NAV actions when running webarena and miniwob benchmarks
USE_CONCISE_ANSWER = (
    os.environ.get('USE_CONCISE_ANSWER', 'false') == 'true'
)  # only return concise answer when running webarena and miniwob benchmarks

if not USE_NAV and USE_CONCISE_ANSWER:
    EVAL_MODE = True  # disabled NAV actions and only return concise answer, for webarena and miniwob benchmarks\
else:
    EVAL_MODE = False


def get_error_prefix(last_browser_action: str) -> str:
    return f'IMPORTANT! Last action is incorrect:\n{last_browser_action}\nThink again with the current observation of the page.\n'


def get_system_message(goal: str, action_space: str) -> str:
    current_datetime = datetime.now().strftime('%a, %b %d, %Y %H:%M:%S')
    
    return f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Use Google Flights for questions \
related to flight search. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

# Goal:
{goal}

# Action Space
{action_space}

# Current Date and Time:
{current_datetime}
"""


CONCISE_INSTRUCTION = """\

Here is another example with chain of thought of a valid action when providing a concise answer to user:
"
In order to accomplish my goal I need to send the information asked back to the user. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I will send a message back to user with the answer.
```send_msg_to_user("$279.49")```
"
"""


def get_prompt(
    error_prefix: str, cur_url: str, cur_axtree_txt: str, prev_action_str: str
) -> str:
    prompt = f"""\
{error_prefix}

# Current Page URL:
{cur_url}

# Current Accessibility Tree:
{cur_axtree_txt}

# Previous Actions
{prev_action_str}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"
""".strip()
    if USE_CONCISE_ANSWER:
        prompt += CONCISE_INSTRUCTION
    return prompt


class BrowsingAgent():
    def __init__(self, 
                 llm: LLM,
                 logger: Logger,
                 **kwargs):
        self.llm = llm
        self.logger = logger
        
        action_subsets = ['chat', 'bid']
        if USE_NAV:
            action_subsets.append('nav')
        self.action_space = HighLevelActionSet(
            subsets=action_subsets,
            strict=False,  # less strict on the parsing of the actions
            multiaction=True,  # enable to agent to take multiple actions at once
        )
        self.response_parser = BrowsingResponseParser(logger=self.logger)
                
        self.reset()
        
    def reset(self):
        """Resets the Browsing Agent."""
        self.cost_accumulator = 0
        self.error_accumulator = 0
        
        self.prev_actions = []
    
    def step(self, raw_obs):
        """Performs one step using the Browsing Agent.
        This includes gathering information on previous steps and prompting the model to make a browsing command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - BrowseInteractiveAction(browsergym_command) - BrowserGym commands to run
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        # messages: list[Message] = []
        # prev_actions = []
        cur_url = ''
        cur_axtree_txt = ''
        error_prefix = ''
        # last_obs = None
        # last_action = None
        goal = raw_obs['goal']
        
        observation = {
            'clean_axtree_txt': cur_axtree_txt,
            'error_prefix': error_prefix,
            'goal': goal,
        }
        
        step = {
            'observation': observation,
            'state': None,
            'intent': None,
            'action': None,
        }

        prev_action_str = '\n'.join(self.prev_actions)
        # last_action = self.prev_actions[-1]
        
        if raw_obs['last_action_error']:
            error_prefix = get_error_prefix(raw_obs['last_action'])
            observation.update({'error_prefix': error_prefix})
            
            self.error_accumulator += 1
            if self.error_accumulator > 5:
                action = "send_msg_to_user('Too many errors encountered. Task failed.')"
                step.update({'action': action})
                return action, step
            
        cur_url = raw_obs['url']
        try:
            cur_axtree_txt = flatten_axtree_to_str(
                raw_obs['axtree_object'],
                extra_properties=raw_obs['extra_element_properties'],
                with_clickable=True,
                filter_visible_only=True,
            )
            obs_txt = cur_url + '\n' + cur_axtree_txt
            observation.update({'clean_axtree_txt': obs_txt})
            
            self.logger.info(f'*Observation*: {obs_txt}')
            
        except Exception as e:
            self.logger.error(
                'Error when trying to process the accessibility tree: %s', e
            )
            action = "send_msg_to_user('Error encountered when browsing.')"
            step.update({'action': action})
            return action, step
            # return {'return_action': "send_msg_to_user('Error encountered when browsing.')"}
        
        
        system_msg = get_system_message(
            goal,
            self.action_space.describe(with_long_description=False, with_examples=True),
        )
        prompt = get_prompt(error_prefix, cur_url, cur_axtree_txt, prev_action_str)
        
        messages = [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm.completion(
            messages=messages,
            stop=[')```', ')\n```'],
        )
        
        parser_output = self.response_parser.parse(response)
        thought, action = parser_output['thought'], parser_output['action']
        step.update({'state': thought,
                     'action': action})
        
        self.prev_actions.append(action)
        
        self.logger.info(f'*Thought*: {thought}')
        self.logger.info(f'*Action*: {action}')

        return action, step
    

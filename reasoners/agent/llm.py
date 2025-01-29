from abc import abstractmethod

from functools import partial
import os
import time
import traceback
from typing import TYPE_CHECKING, Callable, Optional, Tuple, List, Union
from logging import Logger

from .utils import ParseError, parse_html_tags_raise

if TYPE_CHECKING: 
    from easyweb.llm.llm import LLM as EasyWebLLM

class LLM:
    @abstractmethod
    def __call__(self, *args, **kargs): ...
    
    
# DEBUG
# LOG_FOLDER = '../ama-logs'

def identity(x):
    return x, True, None

def parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ''

class EasyWebParserLLM(LLM):
    def __init__(
        self,
        easyweb_llm: 'EasyWebLLM',
        keys: Union[List[str], Tuple[str]] = (),
        optional_keys: Union[List[str], Tuple[str]] = (),
        logger: Logger = None,
        max_retries: int = 4,
    ):
        super().__init__()
        self.easyweb_llm = easyweb_llm
        self.logger = logger
        self.keys = keys
        self.optional_keys = optional_keys
        if not self.keys and not self.optional_keys: 
            self.parser = identity
        else:
            self.parser = partial(parser, keys=keys, optional_keys=optional_keys)
        self.max_retries = max_retries
        self.cost_accumulator = 0
        
    def completion(self, *args, **kwargs): 
        return self.easyweb_llm.completion(*args, **kwargs)

    def __call__(self, user_prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            ans_dict = self._retry(
                messages, self.parser, n_retries=self.max_retries, **kwargs
            )
            ans_dict['n_retry'] = (len(messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {}
            ans_dict['err_msg'] = str(e)
            ans_dict['stack_trace'] = traceback.format_exc()
            ans_dict['n_retries'] = self.max_retries

        ans_dict['messages'] = messages
        ans_dict['prompt'] = user_prompt

        # DEBUG
        LOG_FOLDER = os.environ.get('DEBUG_LOG_FOLDER', None)
        if LOG_FOLDER and os.path.isdir(LOG_FOLDER):
            with open(f'{LOG_FOLDER}/{str(int(time.time()))}.log', 'w') as f:
                for m in messages:
                    f.write(f"{m['role']}\n\n{m['content']}\n\n\n")

        return ans_dict

    def _retry(
        self,
        messages,
        parser,
        n_retries=4,
        min_retry_wait_time=60,
        rate_limit_max_wait_time=60 * 30,
        **kwargs,
    ):
        tries = 0
        rate_limit_total_delay = 0
        while tries < n_retries and rate_limit_total_delay < rate_limit_max_wait_time:
            response = self.completion(
                messages=messages,
                **kwargs,
            )
            answer = response['choices'][0]['message']['content'].strip()

            messages.append({'role': 'assistant', 'content': answer})

            value, valid, retry_message = parser(answer)
            if valid:
                self.log_cost(response)
                return value

            tries += 1
            msg = f'Query failed. Retrying {tries}/{n_retries}.\n[LLM]:\n{answer}\n[User]:\n{retry_message}'
            if self.logger: 
                self.logger.info(msg)
            messages.append({'role': 'user', 'content': retry_message})

        raise ValueError(f'Could not parse a valid value after {n_retries} retries.')

    def log_cost(self, response):
        # TODO: refactor to unified cost tracking
        try:
            cur_cost = self.easyweb_llm.completion_cost(response)
        except Exception:
            cur_cost = 0
        self.cost_accumulator += cur_cost
        if self.logger:
            self.logger.info(
                'Cost: %.2f USD | Accumulated Cost: %.2f USD',
                cur_cost,
                self.cost_accumulator,
            )
            
    
class EasyWebParserMultiResponseLLM(EasyWebParserLLM):
    def __call__(self, user_prompt, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            ans_dicts = self._retry(
                messages, self.parser, n_retries=self.max_retries, **kwargs
            )
            ans_dict = {'answers': ans_dicts}
            ans_dict['n_retry'] = (len(messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {}
            ans_dict['err_msg'] = str(e)
            ans_dict['stack_trace'] = traceback.format_exc()
            ans_dict['n_retries'] = self.max_retries

        ans_dict['messages'] = messages
        ans_dict['prompt'] = user_prompt

        return ans_dict

    def _retry(
        self,
        messages,
        parser,
        n_retries=4,
        min_retry_wait_time=60,
        rate_limit_max_wait_time=60 * 30,
        n=1,
        **kwargs,
    ):
        output_values = []
        tries = 0
        rate_limit_total_delay = 0
        while tries < n_retries and rate_limit_total_delay < rate_limit_max_wait_time:
            remaining_n = n - len(output_values)
            response = self.completion(
                messages=messages,
                n=remaining_n,
                **kwargs,
            )
            answers = [c['message']['content'].strip() for c in response['choices']]
            
            self.log_cost(response)
            outputs = [parser(answer) for answer in answers]
            invalid_answer = None
            invalid_retry_message = None
            for answer, (value, valid, retry_message) in zip(answers, outputs):
                if valid:
                    output_values.append(value)
                    if len(output_values) == n:
                        # self.log_cost(response)
                        return output_values
                else:
                    invalid_answer = value
                    invalid_retry_message = retry_message

            tries += 1
            msg = f'Query failed. Retrying {tries}/{n_retries}.\n[LLM]:\n{invalid_answer}\n[User]:\n{invalid_retry_message}'
            if self.logger: 
                self.logger.info(msg)

        raise ValueError(f'Could not parse a valid value after {n_retries} retries.')
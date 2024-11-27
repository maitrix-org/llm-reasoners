import os
import time
import traceback
from abc import abstractmethod
from typing import Callable, Optional, Tuple
from logging import Logger

# DEBUG
LOG_FOLDER = '../ama-logs'
# LOG_FOLDER = None


class BaseLLM:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kargs): ...


def IDENTITY(x):
    return x, True, None


class OpenDevinParserLLM(BaseLLM):
    def __init__(
        self,
        opendevin_llm: object,
        logger: Logger = None,
        max_retries: int = 4,
        default_parser: Callable[[str], Tuple[str, bool, Optional[str]]] = IDENTITY,
    ):
        super().__init__()
        self.opendevin_llm = opendevin_llm
        self.logger = logger
        self.default_parser = default_parser
        self.max_retries = max_retries
        self.cost_accumulator = 0

    def __call__(self, user_prompt, system_prompt=None, parser=None, **kwargs):
        if parser is None:
            parser = self.default_parser
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            ans_dict = self._retry(
                messages, parser, n_retries=self.max_retries, **kwargs
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
            response = self.opendevin_llm.completion(
                messages=messages,
                # messages=truncated_messages,  # added
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
            cur_cost = self.opendevin_llm.completion_cost(response)
        except Exception:
            cur_cost = 0
        self.cost_accumulator += cur_cost
        if self.logger:
            self.logger.info(
                'Cost: %.2f USD | Accumulated Cost: %.2f USD',
                cur_cost,
                self.cost_accumulator,
            )


class OpenDevinParserMultiResponseLLM(OpenDevinParserLLM):
    def __call__(self, user_prompt, system_prompt=None, parser=None, **kwargs):
        if parser is None:
            parser = self.default_parser
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            ans_dicts = self._retry(
                messages, parser, n_retries=self.max_retries, **kwargs
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
            response = self.opendevin_llm.completion(
                messages=messages,
                # messages=truncated_messages,  # added
                n=remaining_n,
                **kwargs,
            )
            answers = [c['message']['content'].strip() for c in response['choices']]
            # answer = response['choices'][0]['message']['content'].strip()

            # messages.append({'role': 'assistant', 'content': answer})

            # value, valid, retry_message = parser(answer)
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
            # if valid:
            #     self.log_cost(response)
            #     return value

            tries += 1
            msg = f'Query failed. Retrying {tries}/{n_retries}.\n[LLM]:\n{invalid_answer}\n[User]:\n{invalid_retry_message}'
            if self.logger: 
                self.logger.info(msg)
            # messages.append({'role': 'user', 'content': retry_message})

        raise ValueError(f'Could not parse a valid value after {n_retries} retries.')

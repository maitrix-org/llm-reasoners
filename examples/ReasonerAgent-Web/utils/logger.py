import logging
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Literal, Mapping

from termcolor import colored

from .config import config

DISABLE_COLOR_PRINTING = config.disable_color

ColorType = Literal[
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'light_grey',
    'dark_grey',
    'light_red',
    'light_green',
    'light_yellow',
    'light_blue',
    'light_magenta',
    'light_cyan',
    'white',
]

LOG_COLORS: Mapping[str, ColorType] = {
    'BACKGROUND LOG': 'blue',
    'ACTION': 'green',
    'OBSERVATION': 'yellow',
    'DETAIL': 'cyan',
    'ERROR': 'red',
    'PLAN': 'light_magenta',
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg_type = record.__dict__.get('msg_type', None)
        if msg_type in LOG_COLORS and not DISABLE_COLOR_PRINTING:
            msg_type_color = colored(msg_type, LOG_COLORS[msg_type])
            msg = colored(record.msg, LOG_COLORS[msg_type])
            time_str = colored(
                self.formatTime(record, self.datefmt), LOG_COLORS[msg_type]
            )
            name_str = colored(record.name, LOG_COLORS[msg_type])
            level_str = colored(record.levelname, LOG_COLORS[msg_type])
            if msg_type in ['ERROR']:
                return f'{time_str} - {name_str}:{level_str}: {record.filename}:{record.lineno}\n{msg_type_color}\n{msg}'
            return f'{time_str} - {msg_type_color}\n{msg}'
        elif msg_type == 'STEP':
            msg = '\n\n==============\n' + record.msg + '\n'
            return f'{msg}'
        return super().format(record)


console_formatter = ColoredFormatter(
    '\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)

file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)
llm_formatter = logging.Formatter('%(message)s')


class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        # start with attributes
        sensitive_patterns = [
            'api_key',
            'aws_access_key_id',
            'aws_secret_access_key',
            'e2b_api_key',
            'github_token',
        ]

        # add env var names
        env_vars = [attr.upper() for attr in sensitive_patterns]
        sensitive_patterns.extend(env_vars)

        # and some special cases
        sensitive_patterns.append('LLM_API_KEY')
        sensitive_patterns.append('SANDBOX_ENV_GITHUB_TOKEN')

        # this also formats the message with % args
        msg = record.getMessage()
        record.args = ()

        for attr in sensitive_patterns:
            pattern = rf"{attr}='?([\w-]+)'?"
            msg = re.sub(pattern, f"{attr}='******'", msg)

        # passed with msg
        record.msg = msg
        return True


def get_console_handler():
    """
    Returns a console handler for logging.
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    return console_handler


def get_file_handler(log_dir=None):
    """
    Returns a file handler for logging.
    """
    log_dir = os.path.join(os.getcwd(), 'logs') if log_dir is None else log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d')
    file_name = f'opendevin_{timestamp}.log'
    file_handler = logging.FileHandler(os.path.join(log_dir, file_name))
    if config.debug:
        file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    return file_handler


# Set up logging
logging.basicConfig(level=logging.ERROR)


def log_uncaught_exceptions(ex_cls, ex, tb):
    """
    Logs uncaught exceptions along with the traceback.

    Args:
        ex_cls (type): The type of the exception.
        ex (Exception): The exception instance.
        tb (traceback): The traceback object.

    Returns:
        None
    """
    logging.error(''.join(traceback.format_tb(tb)))
    logging.error('{0}: {1}'.format(ex_cls, ex))


sys.excepthook = log_uncaught_exceptions

reasoners_logger = logging.getLogger('reasoners_logger')
reasoners_logger.setLevel(logging.INFO)
reasoners_logger.addHandler(get_file_handler())
reasoners_logger.addHandler(get_console_handler())
reasoners_logger.addFilter(SensitiveDataFilter(reasoners_logger.name))
reasoners_logger.propagate = False
reasoners_logger.debug('Logging initialized')
reasoners_logger.debug(
    'Logging to %s', os.path.join(os.getcwd(), 'logs', 'agent_model.log')
)

# Exclude LiteLLM from logging output
logging.getLogger('LiteLLM').disabled = True
logging.getLogger('LiteLLM Router').disabled = True
logging.getLogger('LiteLLM Proxy').disabled = True

def get_agent_logger(log_file='default_log.log', log_dir=None): 
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_file_path, '..', 'logs') if log_dir is None else log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger_name = f"agent_{log_file.replace('.log', '')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Set log file
    ## Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    ## Set up file handler with the specified log file
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.addHandler(get_console_handler())
    logger.addFilter(SensitiveDataFilter(logger.name))
    logger.propagate = False
    logger.debug('Logging initialized')
    logger.debug('Logging to %s', os.path.join(log_dir, log_file))
    return logger


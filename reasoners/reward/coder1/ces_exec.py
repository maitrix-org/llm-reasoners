from traceback import format_exc
import requests
import time
import json
import random
from typing import Tuple

from .utils import check_executor_alive, _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS


# https://github.com/cassanof/code_exec_server
def remote_code_exec_ces(code: str, stdin: str = None) -> Tuple[bool, str]:
    _DEFAULT_CES_URL = "http://localhost:8000"
    timeout = _DEFAULT_TIMEOUT_SECONDS
    max_retry_on_timeout: int = 2
    cur_retry_on_timeout = 0
    while True:  # loop for server downtime
        try:
            t_start = time.time()
            headers = {"Content-Type": "application/json"}
            r = requests.post(
                _DEFAULT_CES_URL + "/py_exec",
                data=json.dumps({
                    "code": code,
                    "timeout": timeout,
                    "stdin": stdin
                }),
                headers=headers,
            )
            resp, outs = r.text.split("\n", 1)
            succ_exit = resp == "0"

            # sometimes the code fail without output despite it should
            # it could because the server is busy and it hits the timeout
            # for such case we should retry.
            # FIXME: also mark this as a potential bug in code exec server or prime
            if not succ_exit and outs == "" and time.time() - t_start > timeout:  # likely server timeout
                note_message = f"🚨 Failed with timeout for {timeout}s X {max_retry_on_timeout} times"
                if cur_retry_on_timeout >= max_retry_on_timeout:
                    return False, _ERROR_MSG_PREFIX + note_message
                cur_retry_on_timeout += 1
                print(note_message)
                time.sleep(random.randint(5, 20))  # sleep randomly to avoid all clients retry at the same time
                continue
            return succ_exit, outs
        except Exception:
            if not check_executor_alive(_DEFAULT_CES_URL):  # check if the server is alive
                print("Request rejected, waiting 3 seconds and then retrying...")
                time.sleep(3)
                continue

            return False, _ERROR_MSG_PREFIX + format_exc()

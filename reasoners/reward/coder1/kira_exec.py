from traceback import format_exc
import requests
import time
import json
from typing import Tuple

from .utils import check_executor_alive, _ERROR_MSG_PREFIX


# Modified version of https://github.com/FlorianWoelki/kira
def remote_code_exec_kira(code: str, stdin: str = None) -> Tuple[bool, str]:
    """Returns either (True, output) or (False, error)"""
    _DEFAULT_KIRA_URL = "http://localhost:9090"
    while True:  # loop for server downtime
        try:
            headers = {"Content-Type": "application/json"}
            r = requests.post(
                _DEFAULT_KIRA_URL + "/execute",
                data=json.dumps({
                    "language": "python",
                    "content": code,
                    "tests": [{
                        "stdin": stdin
                    }]
                }),
                headers=headers,
            )
            response = r.json()
            result = response["testOutput"]["results"]
            if not result:
                return False, _ERROR_MSG_PREFIX + f"{response}"
            result = result[0]
            stdout = result["received"]
            stderr = result["runError"]
            if len(stderr) == 0:
                return True, stdout
            return False, stderr
        except Exception:
            if not check_executor_alive(_DEFAULT_KIRA_URL):  # check if the server is alive
                print("Request rejected, waiting 3 seconds and then retrying...")
                time.sleep(3)
                continue

            return False, _ERROR_MSG_PREFIX + format_exc()

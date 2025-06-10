import requests

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30


def check_executor_alive(executor):
    try:
        return requests.get(executor + "/").status_code in [200, 404]
    except Exception:
        return False

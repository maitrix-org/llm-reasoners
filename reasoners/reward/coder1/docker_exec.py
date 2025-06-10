from traceback import format_exc

import docker
from docker.types import Ulimit
import tempfile
from pathlib import Path

from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

CLI_ARG_SIZE_LIMIT = 1024 * 3


# trying the fifth option as firejail can be tricky to configure in some other test beds
def code_exec_docker(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS):
    image = "jupyter/scipy-notebook"  # python 3.11
    ulimits = [
        Ulimit(name="nofile", soft=16, hard=16),
        Ulimit(name="fsize", soft=524288, hard=524288),  # 512 KB
        Ulimit(name="as", soft=4096 * 1024 * 1024, hard=4096 * 1024 * 1024)  # 4096 MB in bytes RAM
    ]

    client = docker.from_env()
    container_kwargs = {
        "image": image,
        "environment": {
            "OPENBLAS_NUM_THREADS": "1"
        },
        "ulimits": ulimits,
        "stdin_open": stdin is not None,
        "pids_limit": 16,
    }

    tmpdir = None
    if len(code) < CLI_ARG_SIZE_LIMIT:
        pycmd = ["python", "-c", code]
    else:
        tmpdir = tempfile.TemporaryDirectory()
        container_kwargs["volumes"] = {tmpdir.name: {"bind": "/tmp"}}
        script_path = Path(tmpdir.name) / "script.py"
        script_path.write_text(code)
        pycmd = ["python", "/tmp/script.py"]

    if stdin:
        bash_cmd = ["sh", "-c"]
        if "-c" in pycmd:
            pycmd[-1] = f"'{pycmd[-1]}'"
        if len(stdin) < CLI_ARG_SIZE_LIMIT:
            bash_cmd.append(f"echo {repr(stdin)} | {' '.join(pycmd)}")
        else:
            if tmpdir is None:
                tmpdir = tempfile.TemporaryDirectory()
                container_kwargs["volumes"][stdin_path.parent] = {"bind": "/tmp"}

            stdin_path = Path(tmpdir.name) / "stdin.txt"
            stdin_path.write_text(stdin)
            bash_cmd.append(f"cat /tmp/stdin.txt | {' '.join(pycmd)}")
        container_kwargs["command"] = bash_cmd
    else:
        container_kwargs["command"] = pycmd

    container = None
    try:
        container = client.containers.run(**container_kwargs, detach=True)
        result = container.wait(timeout=timeout)
        exit_code = result.get("StatusCode", 1)
        logs = container.logs(stdout=True, stderr=True).decode()
    except Exception:
        return False, _ERROR_MSG_PREFIX + f"{format_exc()}"
    finally:
        if container:
            container.remove(force=True)
        if tmpdir:
            tmpdir.cleanup()

    if exit_code == 0:
        return True, logs
    else:
        return False, _ERROR_MSG_PREFIX + f"Exit code: {exit_code}\nLogs:\n{logs}"

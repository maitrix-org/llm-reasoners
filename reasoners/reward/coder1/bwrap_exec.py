import os
import subprocess

from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

CLI_ARG_SIZE_LIMIT = 1024 * 3


def code_exec_bwrap(
    code,
    stdin: str = None,
    timeout=5,
    pytest: str = None,
    solution: str = None,
    python_env: str = os.environ.get("CONDA_BIN_PATH", None),
):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]  # avoid importing wrong stuff

    if python_env is None:
        python_executable = "/usr/bin/python3"
    else:
        python_executable = os.path.join(python_env, "python3")

    command = [
        "timeout",
        str(timeout),
        "bwrap",
        "--unshare-all",
        # "--ro-bind", "/usr/bin/python3", "/usr/bin/python3",
        "--ro-bind",
        python_executable,
        "/sandbox/bin/python3",
        "--ro-bind",
        "/usr/lib",
        "/usr/lib",
        "--ro-bind",
        "/usr/lib64",
        "/usr/lib64",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        # "--new-session", # causes issues w. timeout
    ]
    if (
        solution
    ):  # not necessarily a pytest file. test solution is just in a separate file that imports solution
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with solution_file"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(solution)
            command.extend(["--ro-bind", tmpdir, tmpdir])
            command.extend(
                ["/sandbox/bin/python3", os.path.join(tmpdir, "test_solution.py")]
            )
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    elif pytest:  # none of the reference solutions in code-r1-12k are pytest based
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.extend(["--ro-bind", tmpdir, tmpdir])
            command.extend(["/sandbox/bin/python3", "-m", "pytest", tmpdir])
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    else:
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command.extend(["/sandbox/bin/python3", "-c", code])
            result = subprocess.run(
                command,
                input=stdin.encode() if stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
        else:
            with NamedTemporaryFile() as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command.extend(["--ro-bind", tmp.name, tmp.name])
                command.extend(["/sandbox/bin/python3", tmp.name])
                result = subprocess.run(
                    command,
                    input=stdin.encode() if stdin else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                )

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

import os
import subprocess

from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

# So I tried 4 approaches for code execution (after a few all-nighters...):
# 1. _remote_code_exec_ces -- Directly using https://github.com/cassanof/code_exec_server
#       - Is fast but leads to unreasonable false positives of timeouts
#       - I tried to alleviate this by (i) restarting the server frequently; (ii) bigger timeout; (iii) lower concurrency
#       - Still feels 10% false positives and bad concurrency
# 2. _remote_code_exec_kira -- Extending https://github.com/cassanof/code_exec_server to support my format and use some OS features for isolation
#       - Less unreasonable timeouts but the concurrency is very bad, stucking at create temp dirs all the time
# 3. https://e2b.dev/
#       - Concurrency is fine
#       - Does not support STDIN by default - needs some hack to support it
#       - I don't want to pay other servers when I have 192 physical CPUs...
# 4. _code_exec_firejail -- Using firejail (https://github.com/netblue30/firejail)
#       - User space isolation (some ulimit/rlimit features)
#       - Drop-in OS isolation via seccomp (blocking socket, etc.)
#       - Concurrency is the best so far
#       - This is not the safest - but docker is not safe either :L. Looks good enough for my dataset anyways.
# sudo add-apt-repository ppa:deki/firejail
# sudo apt-get update
# sudo apt-get install firejail firejail-profiles

CLI_ARG_SIZE_LIMIT = 1024 * 3


def code_exec_firejail(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"] # avoid importing wrong stuff

    # Build the firejail command with resource limits and cleanup options
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2m",  # Limit file size
        "--rlimit-as=4096m",
        f"--timeout=00:00:{timeout}",
    ]

    if pytest:
        # solution is in {tmpdir}/solution.py
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            # Write the solution to a file
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["python3", "-m", "pytest", tmpdir])
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
            command.extend(["python3", "-c", code])
            result = subprocess.run(command,
                                    input=stdin.encode() if stdin else None,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    env=env,
                                    check=False)
        else:
            with NamedTemporaryFile() as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command.insert(4, f"--whitelist={tmp.name}")
                command.extend(["python3", tmp.name])
                result = subprocess.run(command,
                                        input=stdin.encode() if stdin else None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        env=env,
                                        check=False)

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

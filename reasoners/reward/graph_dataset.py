import re
import random
import ast
import operator
import signal
import contextlib

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def extract_solution(solution_str):

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
        if re.search(r'^[A-Za-z]+$', final_answer):
            return final_answer
    else:
        return None

def compute_score(data_source: str,
                  solution_str: str,
                  ground_truth: str,
                  extra_info: any = None,
                  compressed: bool = False,
                  timeout: float = 10.0):
    """The scoring function for graph dataset task.
    
    Args:
        solution_str: the solution text
        ground_truth: the correct answer
        timeout: maximum time in seconds to allow for computation
    """
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]  
        print(f"ground_truth: {ground_truth}")
    try:
        with time_limit(timeout):
            if not isinstance(ground_truth, str):
                ground_truth = str(ground_truth)

            target = ground_truth.lower()
            solution = extract_solution(solution_str)
            
            if solution:
                solution = solution.lower()
            else:
                score = 0.0

            try:
                if target == solution:
                    score = 1.0
                else:
                    score = 0.0

            except Exception as e:
                score = 0.0

    except TimeoutException:
        print("Computation timed out in graph_dataset")
        score = 0.0 
    except Exception as e:
        print(f"Error in compute_score in graph_dataset: {e}")
        score = 0.0

    return float(score)

import re
import random
import ast
import operator
import json
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
        try:
            solution = ast.literal_eval(final_answer)
            return solution
        except (SyntaxError, ValueError):
            try:
                solution = json.loads(final_answer)
                return solution
            except json.JSONDecodeError:
                return None
        except Exception as e:
            print(f"Error extracting solution: {e}")
            return None
    else:
        return None


def compute_accuracy(answer, ground_truth):
    """
    compare grid level accuracy of the final answer w the ground truth
    """
    if not isinstance(answer, dict):
        return 0
    
    # num_objects
    num_rows = len(ground_truth["rows"])
    #num_attributes
    num_cols = len(ground_truth["header"])

    #total_correct_cells
    correct_cells = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if answer["rows"][i][j] == ground_truth["rows"][i][j]:
                correct_cells += 1
    #accuracy
    accuracy = correct_cells / (num_rows * num_cols)
    return accuracy

def compute_score(data_source: str,
                  solution_str: str,
                  ground_truth: str,
                  extra_info: any = None,
                  compressed: bool = False,
                  method: str = 'strict',
                  timeout: float = 10.0):
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]
        print(f"ground_truth: {ground_truth}")
    try:
        with time_limit(timeout):
            predicted_arrangement = extract_solution(solution_str)

            if predicted_arrangement is None:
                score = 0.0 
            else:
                try:
                    accuracy = compute_accuracy(predicted_arrangement, ground_truth)
                    score = accuracy
                except Exception as e:
                    score = 0.0

    except TimeoutException:
        print("Computation timed out in zebra_puzzle")
        score = 0.0
    except Exception as e:
        print(f"Error in compute_score in zebra_puzzle: {e}")
        score = 0.0

    return float(score)

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

    # Find the answer tag content
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    
    if matches:
        final_answer = matches[-1].group(1).strip()
        
        # Use regex to safely extract the bird names without eval
        bird_pattern = r'\[[\s\'\"]*([^\],]+)[\s\'\"]*(?:,[\s\'\"]*([^\],]+)[\s\'\"]*)*\]'
        bird_match = re.search(bird_pattern, final_answer)
        
        if bird_match:
            # Extract all bird names from the list
            bird_list_str = bird_match.group(0)
            # Extract individual birds, handling different formats
            birds = re.findall(r'[\'"]?([\w\s]+)[\'"]?', bird_list_str)
            # Clean up the extracted birds (removing list brackets, quotes, etc.)
            birds = [bird.lower().strip() for bird in birds if bird.strip() and bird.strip() not in ['[', ']']]
            return birds
        else:
            # If we can't extract using regex, try a safer approach
            try:
                # Add quotes around unquoted words to make it valid Python
                fixed_str = re.sub(r'(\[|\s*,\s*)(\w+)(\s*,|\])', r'\1"\2"\3', final_answer)
                return eval(fixed_str)
            except:
                # Last resort: just return as text
                return None
    else:
        return None


def compute_edit_distance(list1, list2):
    """
    Calculate edit distance between two lists.
    Returns the minimum number of operations (insertions, deletions, substitutions)
    required to transform list1 into list2.
    """
    # Create a matrix of size (len(list1)+1) x (len(list2)+1)
    dp = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]
    
    # Initialize the first row and column
    for i in range(len(list1) + 1):
        dp[i][0] = i
    for j in range(len(list2) + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                    dp[i][j-1],      # insertion
                                    dp[i-1][j-1])    # substitution
    
    return dp[len(list1)][len(list2)]

# granular reward function
def compute_score(data_source: str,
                  solution_str: str,
                  ground_truth: str,
                  extra_info: any = None,
                  compressed: bool = False,
                  method: str = 'strict',
                  timeout: float = 10.0):
    """The scoring function for bird puzzles task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        timeout: maximum time in seconds to allow for computation
    """
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]
        print(f"ground_truth: {ground_truth}")

    try:
        with time_limit(timeout):
            target = ground_truth.tolist() if not isinstance(ground_truth,list) else ground_truth
            predicted_arrangement = extract_solution(solution_str=solution_str)
            
            if predicted_arrangement is None:
                score = 0.0

            # Evaluate equation
            try:
                if isinstance(predicted_arrangement, list) and isinstance(target, list):            
                    edit_distance = compute_edit_distance(predicted_arrangement, target)
                    max_possible_dist = max(len(predicted_arrangement), len(target))
                result = predicted_arrangement == target
                if result:
                    score = 1.0
                elif method != 'strict':
                    score = max(1.0 - (edit_distance / max_possible_dist))
                else:
                    score = 0.0
            except Exception as e:
                score = 0.0

    except TimeoutException:
        print("Computation timed out in puzzles_dataset")
        score = 0.0
    except Exception as e:
        print(f"Error in compute_score in puzzles_dataset: {e}")
        score = 0.0

    return float(score)

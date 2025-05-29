from verl.utils.reward_score.prime_math.grader import math_equal
from verl.utils.reward_score import math
import re


def _check_single_answer(answer: str, ground_truth: str) -> bool:
    try:
        nanswer = answer.replace(",", "").replace("%", " / 100").replace("$", "").replace(":", "/").replace("\\", "")
        nanswer = float(eval(nanswer))
        return math_equal(nanswer, ground_truth, tolerance=1e-3)
    except:
        # If the answer is not a number, use the original answer for full string match
        return math.is_equiv(answer, ground_truth)

def drop_latex_text(answer: str) -> str:
    # Remove \\text{} from "20 \\text{to} 39". There could be multiple \\text{} in the answer.
    # Replace \text{something} with something
    answer = re.sub(r'\\\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\\\', r'', answer)
    return answer
    

def compute_score(data_source: str, model_output: str, ground_truth: str, extra_info: any = None, compressed: bool = False) -> float:
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]
        print(f"ground_truth: {ground_truth}")
    
    model_output = str(model_output).lower()
    ground_truth = str(ground_truth).lower()
    
    solution_str = model_output.split("</think>")[-1]
    answer_str = math.last_boxed_only_string(solution_str)
    if answer_str is not None:
        answer = math.remove_boxed(answer_str)
        answer = drop_latex_text(answer)
    else:
        answer = solution_str

    # print(f">>> {answer}, {ground_truth}")
    if "|" not in ground_truth:
        # Single numeric answer
        score = _check_single_answer(answer, ground_truth)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth = sorted([ans.strip() for ans in ground_truth.split("|")])
            answer = sorted([ans.strip() for ans in answer.split("|")])
            if len(ground_truth) != len(answer):
                score = 0
            else:
                score = 1
                for gt, res in zip(ground_truth, answer):
                    score = _check_single_answer(res, gt)
                    if not score:
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            return 0.0

    return float(score)

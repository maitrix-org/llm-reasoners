import re
import ast
import numpy as np

def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, flags=re.DOTALL))
    if matches:
        final_answer = matches[-1].group(1).strip()
        final_answer = final_answer.replace("\n", "")
        final_answer = final_answer.replace("...", "-1")
        try:
            # Find the part of the text that looks like a nested list
            start = final_answer.index('[[')
            end = final_answer.index(']]', start) + 2
            array_str = final_answer[start:end]
            # Use ast.literal_eval to safely evaluate the string as a Python expression
            array = ast.literal_eval(array_str)
            if all(isinstance(i, list) for i in array):
                return array
            else:
                return [[0]]
        except Exception as e:
            return [[0]]
    else:
        return [[0]]

def pad_array_with_value(array, target_shape, pad_value):
    """
    Pad the given array to the target shape with the specified pad value.

    This function pads the original array to fit the target shape by adding additional
    pixels at the ends. This method ensures that the smaller array is placed at the
    top-left corner of the target shape, making sense of the number of correct pixels
    during comparison.

    Note:
    Depending on how you pad the arrays, the number of correct pixels might vary.
    For example, placing the smaller array in the center versus adding pixels at the ends
    can yield different results. Here, we pad by adding pixels at the ends.

    Parameters:
    array (list): The original array to be padded.
    target_shape (tuple): The desired shape of the padded array (rows, columns).
    pad_value (int): The value to use for padding the array.

    Returns:
    np.ndarray: A padded array with the specified target shape and pad value.
    """
    padded_array = np.full(target_shape, pad_value, dtype=int)
    try:
        array = np.stack(array).astype(int)
    except Exception as e:
        array = np.array([[0]])
    original_shape = array.shape
    padded_array[:original_shape[0], :original_shape[1]] = array
    return padded_array


def compare_solutions_with_padding(generated_output, correct_output, pad_value=-1):
    """
    Compare the generated output with the correct output, using padding to align their shapes.

    Parameters:
    generated_output (list): The generated solution array.
    correct_output (list): The correct solution array.
    pad_value (int, optional): The value to use for padding. Default is -1. The colour value -1 should not be present in the solutions.

    Returns:
    tuple: A tuple containing:
        - is_correct (bool): True if the solutions match exactly, False otherwise.
        - correct_percentage (float): The percentage of correctly matched pixels.
    """
    max_rows = max(len(generated_output), len(correct_output))
    max_cols = max(len(generated_output[0]), len(correct_output[0]))
    target_shape = (max_rows, max_cols)
    
    padded_generated = pad_array_with_value(generated_output, target_shape, pad_value)
    padded_correct = pad_array_with_value(correct_output, target_shape, pad_value)
    total_pixels = max_rows * max_cols
    correct_pixels = np.sum((padded_generated == padded_correct) & (padded_generated != pad_value) & (padded_correct != pad_value))
    correct_percentage = (correct_pixels / total_pixels)
    is_correct = float(correct_pixels == total_pixels)
    return is_correct, correct_percentage



def compute_score(data_source: str,
                  model_output: str,
                  ground_truth: np.ndarray,
                  extra_info: any = None,
                  compressed: bool = False) -> float:
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]
        print(f"ground_truth: {ground_truth}")
    model_output = str(model_output)
    final_answer = extract_solution(model_output)
    is_correct, correct_percentage = compare_solutions_with_padding(final_answer, ground_truth)
    return float(is_correct)
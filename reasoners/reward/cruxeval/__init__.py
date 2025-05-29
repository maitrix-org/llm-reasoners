import re
from verl.utils.reward_score.cruxeval.utils import check_correctness


def compute_score(data_source: str,
                  model_output: str,
                  ground_truth: str,
                  extra_info: any = None,
                  compressed: bool = False) -> float:
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]

    model_output = str(model_output)
    # print(f">>> {model_output}")
    try:
        if "</think>" in model_output:
            # remove content until </think>
            model_output = re.split(r"</think>", model_output)[1]
        else:
            model_output = model_output
        # remove content between ```python and ```
        model_output = re.split(r"```python", model_output)[1]
        model_output = re.split(r"```", model_output)[0]
    except:
        model_output = model_output

    full_code = eval(ground_truth)["functional"] + "\n" + model_output
    # print(f">>> {full_code}")
    is_correct = 1 if check_correctness(full_code) else 0
    # print(f">>> {is_correct}")
    return float(is_correct)

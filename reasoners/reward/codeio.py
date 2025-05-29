import json
import ast
import re
from typing import Dict, Any, Tuple, List

def normalize(obj: Any) -> Any:
    """
    Recursively normalize objects so that semantically‑equivalent
    values compare equal.

    Handles:
      • "true"/"false" (any case) → bool
      • strings that are JSON objects/arrays → parsed & normalized
      • lists / dicts → element‑wise normalize
    """
    # ---------- primitives ---------- #
    if isinstance(obj, str):
        s = obj.strip()

        # try JSON string first
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                return normalize(parsed)          # recurse on parsed
            except json.JSONDecodeError:
                pass                              # fall through if not valid JSON

        # bool strings
        low = s.lower()
        if low in {"true", "false"}:
            return low == "true"
        return obj

    # ---------- containers ---------- #
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}

    # everything else untouched
    return obj
# ------------------------------------------------------------------ #

def extract_last_complete_json(s: str):
    """
    Extract the last complete JSON object from a string.
    """
    stack, last_json_start, last_json_str = [], None, None
    for i, ch in enumerate(s):
        if ch == "{":
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    last_json_str = s[last_json_start:i+1]
                    last_json_start = None
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None

def extract_json_from_code_block(text: str):
    """
    Extract JSON content from text that might contain code blocks.
    """
    text = str(text)
    m = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    return (m[0] if m else text).strip()

def extract_answer(output_str: str):
    """
    Extract the predicted answer from the output string.
    """
    if output_str is None:
        return None
    clean = extract_json_from_code_block(output_str)
    try:
        obj = json.loads(clean)
        return obj.get("output", obj.get("input", obj))
    except json.JSONDecodeError:
        last = extract_last_complete_json(clean)
        if last is None:
            return None
        return last.get("output", last.get("input", last))
    except Exception:
        return None

def convert_gt_to_object(gt: str):
    """
    Convert ground truth string to Python objects.
    """
    if isinstance(gt, str) and ('"input":' in gt or '"output":' in gt):
        if not (gt.strip().startswith("{") and gt.strip().endswith("}")):
            gt = "{" + gt + "}"
        try:
            obj = json.loads(gt)
            return obj.get("input", obj.get("output", obj))
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(gt)
        return obj.get("output", obj.get("input", obj))
    except Exception:
        try:
            return ast.literal_eval(gt)
        except Exception:
            return gt

def check_accuracy(raw_pred: str, gt: str, any_order: bool = False) -> Tuple[bool, bool]:
    """
    Check if the prediction matches the ground truth.
    """
    pred = normalize(extract_answer(raw_pred))
    truth = normalize(convert_gt_to_object(gt))

    # top‑level bool convenience shortcut
    tf_groups = [["True", "true", True], ["False", "false", False]]
    for g in tf_groups:
        if pred in g and gt in g:
            return True, False

    no_answer = pred is None
    if not any_order:
        return pred == truth, no_answer

    # order‑agnostic list comparison
    if not isinstance(pred, list) or not isinstance(truth, list):
        return False, no_answer
    if len(pred) != len(truth):
        return False, no_answer
    return all(item in truth for item in pred), no_answer

def compute_score(data_source: str,
                  model_output: str,
                  ground_truth: str,
                  extra_info: any = None,
                  compressed: bool = False) -> float:
    """
    Compute score dict for evaluation harness.
    """
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]
        print(f"ground_truth: {ground_truth}")
    correct, _ = check_accuracy(str(model_output), str(ground_truth), any_order=False)
    return float(correct)


# --------------------------- test --------------------------- #
if __name__ == "__main__":
    # Example 1
    model_out1 = '''```json
{"input": {"upper_limit": 2924}}
```'''
    ground_truth1 = '"input": {"upper_limit": 2719}'
    result1 = compute_score(model_out1, ground_truth1)
    print(f"Example 1 result: {result1}")
    
    # Example 2
    model_out2 = '''```json
{
  "preorder": "[97]",
  "inorder": "[97]"
}
```'''
    ground_truth2 = '"input": {"preorder": "[97]", "inorder": "[97]"}'
    result2 = compute_score(model_out2, ground_truth2)
    print(f"Example 2 result: {result2}")
    
    # Example 3 - testing "output" wrapper
    model_out3 = '''```json
{"output": {"result": 42}}
```'''
    ground_truth3 = '"output": {"result": 42}'
    result3 = compute_score(model_out3, ground_truth3)
    print(f"Example 3 result: {result3}")
    
    # Example 4 - previously failing case, should now pass
    model_out4 = '''```json
{"answer": true}
```'''
    ground_truth4 = '"output": {"answer": "true"}'
    result4 = compute_score(model_out4, ground_truth4)
    print(f"Example 4 result: {result4}")

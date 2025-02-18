import json
import re
import string


def prosqa_extractor(algo_output):
    print(algo_output)
    match = re.search(r"Answer:\s*(.*)", algo_output)
    if match:
        return match.group(1).strip()
    return algo_output

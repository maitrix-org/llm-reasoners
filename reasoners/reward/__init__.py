# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Dict, Any
from reward_score.utils import _deserialise_extra, _decompress_str

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    compressed = True
    reward_metric = None
    if type(ground_truth) is dict:
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]

    if type(extra_info) is dict:
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        reward_metric = extra_info.get("reward_metric", None)
        compressed = False

    # math
    if data_source.startswith("math"):
        if reward_metric == "prime_math":
            from . import prime_math
            res = prime_math.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
        elif reward_metric == "math_llm_judge":
            from . import math_llm_judge
            res = math_llm_judge.compute_score(
                data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed
            )
        else:
            # Default
            from . import naive_dapo
            res = naive_dapo.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    # code generation
    elif data_source.startswith('codegen'):
        from . import coder1
        res = coder1.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    # simulation (code)
    elif data_source.startswith("simulation__codeio"):
        from . import codeio
        res = codeio.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    elif data_source.startswith("simulation__cruxeval"):
        from . import cruxeval
        res = cruxeval.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    # logic
    elif data_source.startswith("simulation__arcagi") or data_source.startswith("simulation__barc"):
        from . import arcagi
        res = arcagi.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    elif data_source.startswith("logic__zebra_puzzle"):
        from . import zebra_puzzle
        res = zebra_puzzle.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    elif data_source.startswith("logic__ordering_puzzle"):
        from . import puzzles_dataset
        res = puzzles_dataset.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    elif data_source.startswith("logic__graph"):
        from . import graph_dataset
        res = graph_dataset.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    # table
    elif data_source.startswith("table"):
        # TODO: tmp placeholder using math_verify
        from . import tablereason
        res = tablereason.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    elif data_source.startswith('stem__gpqa'):
        from . import gpqa
        res = gpqa.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    elif data_source.startswith('stem__supergpqa'):
        from . import supergpqa
        res = supergpqa.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    elif data_source.startswith('stem_web'):
        from . import stem_llm_judge
        res = stem_llm_judge.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info, compressed=compressed)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        if "score" in res:
            res["score"] = float(res["score"])
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])

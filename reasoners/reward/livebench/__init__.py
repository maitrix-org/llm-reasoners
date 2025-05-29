import re
import numpy as np
import ast
from ast import literal_eval
import json

# data analysis split
from verl.utils.reward_score.livebench.data_analysis.tablereformat.utils import table_process_results
from verl.utils.reward_score.livebench.data_analysis.cta.utils import cta_process_results
from verl.utils.reward_score.livebench.data_analysis.tablejoin.utils import joinmap_process_results

# reasoning split
from verl.utils.reward_score.livebench.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
from verl.utils.reward_score.livebench.reasoning.web_of_lies_v3.utils import web_of_lies_v3_process_results
from verl.utils.reward_score.livebench.reasoning.house_traversal.utils import house_traversal_process_results
from verl.utils.reward_score.livebench.reasoning.zebra_puzzle.utils import zebra_puzzle_process_results_old # get_zebra_puzzle_evaluator
from verl.utils.reward_score.livebench.reasoning.spatial.utils import spatial_process_results

# language split
from verl.utils.reward_score.livebench.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from verl.utils.reward_score.livebench.writing.typos.utils import typos_process_results
from verl.utils.reward_score.livebench.writing.connections.utils import connections_process_results_old # get_connections_puzzle_evaluator

# excluded splits
# from livebench.process_results.math.math_competitions.utils import mathcontest_process_results,aime_process_results 
# from livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
# from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results 
# from livebench.process_results.coding.utils import LCB_generation_process_results, code_generation_process_results
# from livebench.process_results.instruction_following.utils import instruction_following_process_results



def compute_score(solution_str, ground_truth, extra_info):
    """
    Compute the score for the solution string.
    """
    score = 0

    # data analysis split
    if extra_info["task"] == "cta":
        score = cta_process_results(ground_truth, solution_str)
    elif extra_info["task"] == "tablejoin":
        score = joinmap_process_results(ground_truth, solution_str)
    elif extra_info["task"] == "tableformat":
        score = table_process_results(ground_truth, solution_str)
    # reasoning split
    elif extra_info["task"] == "web_of_lies_v2":
        score = web_of_lies_process_results(ground_truth, solution_str)
    elif extra_info["task"] == "web_of_lies_v3":
        score = web_of_lies_v3_process_results(ground_truth, solution_str)
    elif extra_info["task"] == "house_traversal":
        score = house_traversal_process_results(ground_truth, solution_str)
    elif extra_info["task"] == "zebra_puzzle":
        score = zebra_puzzle_process_results_old(ground_truth, solution_str)
    elif extra_info["task"] == "spatial":
        score = spatial_process_results(ground_truth, solution_str)
    # language split
    elif extra_info["task"] == "plot_unscrambling":
        score = plot_unscrambling_process_results(ground_truth, solution_str)
    elif extra_info["task"] == "typos":
        score = typos_process_results(ground_truth, solution_str)
    elif extra_info["task"] == "connections":
        score = connections_process_results_old(ground_truth, solution_str)

    return {
        "score": score,
        "acc": score,
    }

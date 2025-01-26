#!/usr/bin/env python3

import argparse
import json
import os
import time
import traceback
import multiprocessing
from functools import partial
from typing import List, Dict, Any, Tuple

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm.auto import tqdm

from reasoners import WorldModel, LanguageModel, SearchConfig, State, Reasoner
from reasoners.algorithm import BeamSearch, MCTS
from reasoners.lm import HFModel, SGLangModel
from reasoners.visualization import visualize
from qwen_math_parser import math_equal

from world_model import MathModel
from search_config import MathConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Math 500 Evaluation Script')
    parser.add_argument('--reward-sglang-url', type=str, default='http://127.0.0.1:30002',
                      help='SGLang API URL (default: http://127.0.0.1:30002/v1)')
    parser.add_argument('--reward-model-path', type=str, required=True,
                      help='Path to the reward model')
    parser.add_argument('--reward-model-device', type=str, default='cuda:0',
                      help='Device to run reward model on (default: cuda:0)')
    parser.add_argument('--prompt-path', type=str, required=True,
                      help='Path to prompts JSON file')
    parser.add_argument('--output-path', type=str, default='answers.json',
                      help='Path to save results (default: answers.json)')
    parser.add_argument('--policy-sglang-url', type=str, default='http://127.0.0.1:30001',
                      help='SGLang API URL (default: http://127.0.0.1:30001/v1)')
    parser.add_argument('--beam-size', type=int, default=2,
                      help='Beam size for search (default: 2)')
    parser.add_argument('--max-depth', type=int, default=40,
                      help='Maximum search depth (default: 40)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for sampling (default: 0.7)')
    parser.add_argument('--log-file', type=str, default='output.log',
                      help='Path to log file (default: output.log)')
    parser.add_argument('--num-processes', type=int, default=4,
                      help='Number of parallel processes (default: 4)')
    parser.add_argument('--num-actions', type=int, default=1,
                        help='Number of actions to consider (default: 1)')
    return parser.parse_args()

def setup_logging(log_file):
    logger.remove()
    logger.add(log_file, enqueue=True)  # enqueue=True for multiprocessing support

def load_models(args):
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["SGLANG_API_URL"] = args.policy_sglang_url

    llm = SGLangModel(
        model="",
        is_instruct_model=True,
        url=args.policy_sglang_url,
    )

    reward_model = SGLangModel(
        model="",
        is_instruct_model=False,
        url=args.reward_sglang_url
    )

    return llm, reward_model

def setup_reasoner(llm, reward_model, prompt, args):
    config = MathConfig(
        base_model=llm,
        prm=reward_model,
        prm_tokenizer_path=args.reward_model_path,
        prompt=prompt,
        num_actions=args.num_actions,
        temperature=args.temperature
    )
    
    search_algorithm = BeamSearch(
        beam_size=args.beam_size,
        max_depth=args.max_depth
    )

    return Reasoner(
        world_model=MathModel(base_model=llm, prompt=prompt, prm=reward_model),
        search_config=config,
        search_algo=search_algorithm,
    )

def eval_answer(ground_truth, predicted):
    ground_truth = ground_truth.replace(" ", "")
    predicted = predicted.replace(" ", "")
    return math_equal(ground_truth, predicted)

def process_chunk(chunk_idx: int, chunk: list[dict], args: argparse.Namespace, prompt: dict):
    answers_to_save = []
    times = []
    
    try:
        llm, reward_model = load_models(args)
        reasoner = setup_reasoner(llm, reward_model, prompt, args)
    except Exception as e:
        logger.error(f"Chunk {chunk_idx} failed initialization: {str(e)}")
        return [], []

    temp_file = f"{args.output_path}.tmp.{chunk_idx}"
    
    for example in chunk:
        start_time = time.time()
        try:
            problem = {"init": example["problem"]}
            trace = reasoner(problem)
            solution = trace.terminal_state.solution
            steps = trace.terminal_state.steps

            is_correct = eval_answer(solution, example["answer"])
            result = {
                "problem": example["problem"],
                "predicted_answer": solution,
                "steps": steps,
                "ground_truth_steps": example["solution"],
                "ground_truth_answer": example["answer"],
                "is_potentially_correct": is_correct,
            }
            answers_to_save.append(result)
            
            with open(temp_file, "w") as f:
                json.dump(answers_to_save, f)

        except Exception as e:
            logger.error(f"Error processing example: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            times.append(time.time() - start_time)
    
    return answers_to_save, times

def main():
    args = parse_args()
    setup_logging(args.log_file)

    with open(args.prompt_path, "r") as f:
        prompt = json.load(f)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset_list = [ex for ex in dataset]

    num_processes = args.num_processes
    chunk_size = len(dataset_list) // num_processes
    chunks = [dataset_list[i*chunk_size : (i+1)*chunk_size] for i in range(num_processes)]
    
    # Distribute remainder examples
    remainder = len(dataset_list) % num_processes
    for i in range(remainder):
        chunks[i].append(dataset_list[num_processes * chunk_size + i])

    logger.info(f"Starting processing with {num_processes} processes")

    # Add progress tracking
    total_examples = len(dataset_list)
    logger.info(f"Total examples to process: {total_examples}")

    chunks_with_indices = list(enumerate(chunks))

    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_chunk, args=args, prompt=prompt)
        results = list(tqdm(
            pool.starmap(process_func, chunks_with_indices),
            total=len(chunks_with_indices),
            desc="Processing chunks"
        ))

    # Aggregate all temporary files
    all_answers = []
    for i in range(num_processes):
        temp_file = f"{args.output_path}.tmp.{i}"
        if os.path.exists(temp_file):
            try:
                with open(temp_file, "r") as f:
                    all_answers.extend(json.load(f))
                os.remove(temp_file)
            except Exception as e:
                logger.error(f"Error loading temp file {temp_file}: {str(e)}")

    # Save final results
    with open(args.output_path, "w") as f:
        json.dump({"answers": all_answers}, f, indent=4)

    avg_time = sum(all_times) / len(all_times)
    logger.info(f"Average time per problem: {avg_time:.2f} seconds")
    logger.info(f"Total time: {sum(all_times):.2f} seconds")
    
    correct = sum(1 for ans in all_answers if ans["is_potentially_correct"])
    accuracy = correct / len(all_answers) if len(all_answers) > 0 else 0
    logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{len(all_answers)})")

if __name__ == "__main__":
    main()
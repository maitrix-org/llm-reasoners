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
    temp_file = f"{args.output_path}.tmp.{chunk_idx}"
    chunk_start = time.time()
    
    try:
        llm, reward_model = load_models(args)
        reasoner = setup_reasoner(llm, reward_model, prompt, args)
    except Exception as e:
        error_msg = f"Chunk {chunk_idx} initialization failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        with open(temp_file, "w") as f:
            json.dump({"error": error_msg, "chunk_idx": chunk_idx}, f, indent=4)
        return [], []

    for idx, example in enumerate(chunk):
        entry = {
            "problem_id": example.get("id", hash(example["problem"])),
            "problem": example["problem"],
            "status": "pending",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "processing_time": None,
            "error": None,
            "traceback": None
        }
        
        start_time = time.time()
        try:
            # Processing logic
            problem = {"init": example["problem"]}
            trace = reasoner(problem)
            solution = trace.terminal_state.solution
            steps = trace.terminal_state.steps

            entry.update({
                "status": "completed",
                "predicted_answer": solution,
                "predicted_steps": steps,
                "ground_truth": {
                    "steps": example["solution"].split("\n"),
                    "answer": example["answer"]
                },
                "is_correct": eval_answer(solution, example["answer"]),
                "processing_time": time.time() - start_time,
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })

        except Exception as e:
            entry.update({
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "processing_time": time.time() - start_time,
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })
            logger.error(f"Chunk {chunk_idx} example {idx} failed: {str(e)}")

        finally:
            # Atomic write with progress tracking
            temp_write = f"{temp_file}.writing"
            try:
                if os.path.exists(temp_file):
                    with open(temp_file, "r") as f:
                        existing = json.load(f)
                else:
                    existing = []
                
                existing.append(entry)
                with open(temp_write, "w") as f:
                    json.dump({
                        "metadata": {
                            "chunk_idx": chunk_idx,
                            "total_examples": len(chunk),
                            "chunk_start": chunk_start,
                            "chunk_duration": time.time() - chunk_start
                        },
                        "results": existing
                    }, f, indent=4)
                
                os.replace(temp_write, temp_file)
            except Exception as e:
                logger.error(f"Failed to write temp file: {str(e)}")

    return answers_to_save, times

def main():
    total_start = time.time()

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

    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            process_func = partial(process_chunk, args=args, prompt=prompt)
            results = list(tqdm(
                pool.starmap(process_func, chunks_with_indices),
                total=len(chunks_with_indices),
                desc="Processing chunks"
            ))
    finally:
        total_duration = time.time() - total_start
        
        # Aggregate all results
        final_output = {
            "metadata": {
                "total_time": total_duration,
                "average_time": total_duration / len(dataset_list) if dataset_list else 0,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(total_start)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "args": vars(args),
                "success_rate": None
            },
            "results": []
        }

        # Load all temp files
        success_count = 0
        for i in range(num_processes):
            temp_file = f"{args.output_path}.tmp.{i}"
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, "r") as f:
                        data = json.load(f)
                        if "results" in data:
                            final_output["results"].extend(data["results"])
                            success_count += len([r for r in data["results"] if r["status"] == "completed"])
                        os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Error loading {temp_file}: {str(e)}")

        # Calculate statistics
        if final_output["results"]:
            times = [r["processing_time"] for r in final_output["results"] if r["processing_time"] is not None]
            final_output["metadata"].update({
                "success_rate": success_count / len(final_output["results"]),
                "time_stats": {
                    "total": sum(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
            })

        with open(args.output_path, "w") as f:
            json.dump(final_output, f, indent=4, sort_keys=True)

        logger.info(f"\nFinal Statistics:\n"
                    f"Total time: {final_output['metadata']['total_time']:.2f}s\n"
                    f"Average time per problem: {final_output['metadata']['time_stats']['average']:.2f}s\n"
                    f"Success rate: {final_output['metadata']['success_rate']:.1%}\n"
                    f"Problems processed: {len(final_output['results'])}/{len(dataset_list)}")

if __name__ == "__main__":
    main()
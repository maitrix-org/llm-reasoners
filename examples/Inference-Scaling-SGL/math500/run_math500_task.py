import asyncio
import argparse
import json
import os
import time
import traceback

from datasets import  load_dataset
from loguru import logger
from tqdm.auto import tqdm

from reasoners import Reasoner
from reasoners.algorithm import BeamSearch, AsyncBeamSearch
from reasoners.lm import HFModel, OpenAIModel
from qwen_math_parser import math_equal

from world_model import MathModel
from search_config import MathConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Math 500 Evaluation Script')
    parser.add_argument('--reward-model-path', type=str, required=True,
                      help='Path to the reward model')
    parser.add_argument('--prompt-path', type=str, required=True,
                      help='Path to prompts JSON file')
    parser.add_argument('--output-path', type=str, default='answers.json',
                      help='Path to save results (default: answers.json)')
    parser.add_argument('--sglang-url', type=str, default='http://127.0.0.1:30001/v1',
                      help='SGLang API URL (default: http://127.0.0.1:30001/v1)')
    parser.add_argument('--reward-model-device', type=str, default='cuda:0',
                      help='Device to run reward model on (default: cuda:0)')
    parser.add_argument('--beam-size', type=int, default=1,
                      help='Beam size for search (default: 1)')
    parser.add_argument('--max-depth', type=int, default=40,
                      help='Maximum search depth (default: 40)')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for sampling (default: 0.7)')
    parser.add_argument('--log-file', type=str, default='output.log',
                      help='Path to log file (default: output.log)')
    parser.add_argument('--max-reasoning-steps', type=int, default=6,
                      help='Maximum reasoning steps (default: 6)')
    parser.add_argument('--num-actions', type=int, default=1,
                        help='Number of actions to consider (default: 1)')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Number of workers for async beam search (default: 8)')
    parser.add_argument('--beam-concurrency', type=int, default=4,
                        help='Beam concurrency for async beam search (default: 4)')
    parser.add_argument('--action-concurrency', type=int, default=8,
                        help='Action concurrency for async beam search (default: 8)')

    return parser.parse_args()

def setup_logging(log_file):
    logger.add(log_file)
    logger.remove(0)

def load_models(args):
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = "dummy"
    os.environ["SGLANG_API_URL"] = args.sglang_url

    # Initialize base model
    llm = OpenAIModel(
        model="model",
        backend="sglang",
        is_instruct_model=True
    )

    # Initialize reward model
    reward_model = HFModel(
        model_pth=args.reward_model_path,
        tokenizer_pth=args.reward_model_path,
        device=args.reward_model_device,
    )

    return llm, reward_model

def setup_reasoner(llm, reward_model, prompt, args):
    config = MathConfig(
        base_model=llm,
        prm=reward_model,
        prompt=prompt,
        num_actions=args.num_actions,
        temperature=args.temperature
    )

    search_algorithm = AsyncBeamSearch(
        beam_size=args.beam_size,
        max_depth=args.max_depth,
        max_workers=args.max_workers ,               # max_workers CPU threads
        beam_concurrency=args.beam_concurrency,      # num beam items processed in parallel
        action_concurrency=args.action_concurrency   # num actions processed in parallel per beam item
    )

    return Reasoner(
        world_model=MathModel(base_model=llm, prm=reward_model, prompt=prompt, max_steps = args.max_reasoning_steps),
        search_config=config,
        search_algo=search_algorithm,
    )

def eval_answer(ground_truth, predicted):
    ground_truth = ground_truth.replace(" ", "")
    predicted = predicted.replace(" ", "")
    return math_equal(ground_truth, predicted)

async def process_dataset(reasoner, dataset, output_path):
    problem_fn = lambda x: {"init": x["problem"]}
    answers_to_save = []
    times = []

    progress_bar = tqdm(total=len(dataset[:3]), desc="Processing problems")
    
    for i in range(len(dataset[:3])):
        logger.info(f"Processing problem {i+1}/{len(dataset)}")
        start = time.time()
        
        try:
            # Get solution
            trace = await reasoner(problem_fn(dataset[i]))
            solution = trace.terminal_state.solution
            steps = trace.terminal_state.steps

            # Prepare results
            result = {
                "problem": dataset[i]["problem"],
                "predicted_answer": solution,
                "steps": steps,
                "ground_truth_steps": dataset[i]["solution"],
                "ground_truth_answer": dataset[i]["answer"],
                "is_potentially_correct": eval_answer(
                    solution,
                    dataset[i]["answer"],
                ),
            }
            answers_to_save.append(result)
            progress_bar.update(1)

        except Exception as e:
            logger.error(f"Error processing problem {i}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
        finally:
            # Record time and save results
            times.append(time.time() - start)
            with open(output_path, "w") as file:
                json.dump({"answers": answers_to_save}, file, indent=4)

    progress_bar.close()

    # Print statistics
    avg_time = sum(times) / len(times)
    logger.info(f"Average time per problem: {avg_time:.2f} seconds")
    logger.info(f"Total time: {sum(times):.2f} seconds")
    
    # Calculate accuracy
    correct = sum(1 for ans in answers_to_save if ans["is_potentially_correct"])
    accuracy = correct / len(answers_to_save)
    logger.info(f"Accuracy: {accuracy:.2%}")

async def main():
    args = parse_args()
    setup_logging(args.log_file)

    llm, reward_model = load_models(args)
    with open(args.prompt_path, "r") as file:
        prompt = json.load(file)

    reasoner = setup_reasoner(llm, reward_model, prompt, args)

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    await process_dataset(reasoner, dataset, args.output_path)

if __name__ == "__main__":
    asyncio.run(main())
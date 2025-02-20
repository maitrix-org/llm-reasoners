import os
import argparse
from utils import prosqa_extractor 
from reasoner import ProsQAReasoner
from reasoners.lm.deepseek_model import DeepseekModel
from reasoners.benchmark.prosqa import ProsQAEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='ProsQA Evaluation Script')
    parser.add_argument('--backend', type=str, default='deepseek', 
                        help='Backend to use for the model (default: deepseek)')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the model')
    parser.add_argument('--max-tokens', type=int, default=None, 
                        help='Maximum number of tokens (default: None)')
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help='Temperature for sampling (default: 0.0)')
    
    return parser.parse_args()


def main():
    args = parse_args()

    model = DeepseekModel(model = args.model_path, backend=args.backend, 
            max_tokens=args.max_tokens, temperature = args.temperature)

    evaluator = ProsQAEvaluator(
        output_extractor=prosqa_extractor, 
        answer_extractor=lambda x: x["answer"]
        )
    reasoner = ProsQAReasoner(model,args.temperature)

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=0)
    print(f'accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()

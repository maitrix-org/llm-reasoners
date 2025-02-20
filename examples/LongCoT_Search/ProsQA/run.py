import os
import argparse
from utils import prosqa_extractor 
from reasoners.lm.deepseek_model import DeepseekModel
from reasoners.benchmark.prosqa import ProsQAEvaluator

class ProsQAReasoner():
    def __init__(self, base_model,temperature=0, bs=1):
        self.base_model = base_model
        self.temperature = temperature
        
    def __call__(self, example, prompt=None):
        outputs = self.base_model.generate(
            [example], 
            temeprature = self.temperature, 
            hide_input=True).text[0]
        return outputs

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

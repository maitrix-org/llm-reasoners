from reasoners import Reasoner, SearchConfig, DefaultWorldModel
from reasoners.algorithm import RandomShooting
from reasoners.algorithm.random import RandomShootingResult
from reasoners.lm import Llama3Model
from reasoners.benchmark import GSM8KEvaluator

from transformers import AutoTokenizer, AutoModel
import torch
import json
import fire

from examples.cot_gsm8k import utils

class EurusSearchConfig(SearchConfig):

    def __init__(self,
                 base_model,
                 temperature=0.7,
                 base_reward_model="openbmb/Eurus-RM-7b"):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        device = torch.device("cuda:1")
        self.reward_model = AutoModel.from_pretrained(base_reward_model, trust_remote_code=True, device_map='cpu')
        self.reward_model.to(device)
        self.reward_model.eval()
        self.reward_model_tokenizer = AutoTokenizer.from_pretrained(base_reward_model)

    def get_actions(self, state):
        # since we only want to do best of N (random shooting)
        # we don't need to generate multiple candidate actions
        # this function will return a random action
        inputs = self.prompt["cot"].replace("{QUESTION}", self.example)
        outputs = self.base_model.generate([inputs],
                                            hide_input=True,
                                            do_sample=True,
                                            max_new_tokens=512,
                                            temperature=self.temperature,
                                            eos_token_id=[".\n\n"]).text[0]
        outputs = outputs.strip() if outputs.strip().endswith(".") else outputs.strip() + '.'
        return [outputs]

    def reward(self, state, action):
        
        template = "[INST] {INST} [/INST] {RESPONSE}"
        inputs = template.replace("{INST}", self.example) \
            .replace("{RESPONSE}", action)
        inputs = self.reward_model_tokenizer(inputs, return_tensors="pt").to(self.reward_model.device)
        reward = self.reward_model(**inputs).item()
        return reward, {}


class EurusWorldModel(DefaultWorldModel):
    def is_terminal(self, state):
        return len(state) == 1

def best_of_n_extractor(result: RandomShootingResult):
    answer = utils.retrieve_answer(result.best_trajectory[-1][1][0])
    return answer

def main(model_dir, llama_size="8B", prompt="examples/cot_gsm8k/prompts/cot.json", best_of_n=10, resume=0, log_dir=None):

    base_model = Llama3Model(model_dir, llama_size, max_batch_size=1)
    
    with open(prompt) as f:
        prompt = json.load(f)

    evaluator = GSM8KEvaluator(
        output_extractor=best_of_n_extractor,
        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
        init_prompt=prompt,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="cot")

    reasoner = Reasoner(
        world_model=EurusWorldModel(base_model),
        search_config=EurusSearchConfig(base_model),
        search_algo=RandomShooting(n_shoot=best_of_n)
    )

    # run the reasoner
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, log_dir=log_dir, resume=resume)

    print(accuracy)

if __name__ == "__main__":
    fire.Fire(main)
    # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 1 examples/Eurus/inference.py --model_dir $LLAMA3_CKPTS --best_of_n 10
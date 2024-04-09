from tqdm import tqdm
from datetime import datetime
import random
from reasoners import Evaluator


def get_cot_prompt(sampled_data):
    formatted_examples = ""
    for i, entry in enumerate(sampled_data, 1):
        formatted_examples += f"Q: {entry['Facts']} {entry['claims'][0]} {entry['Query']}\n"
        formatted_examples += f"A: {entry['claims'][0]} "
        for j, (claim, next_step) in enumerate(zip(entry['claims'][1:], entry['next_steps'][:-1]), 1):
            formatted_examples += f"{next_step} So {claim} "
        tf = not (("not" in entry['claims'][-1]) ^ ("not" in entry['Query']))
        formatted_examples += f"The answer is {'true' if tf else 'false'}.\n\n"
    return formatted_examples

class ProntoQAEvaluatorFinal(Evaluator):
    def __init__(self, 
                 output_extractor= lambda x: x.terminal_state.body if x.terminal_state is not None else "",
                 answer_extractor= lambda x: x.test_example.answer,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot", dataset=None) -> None:

        dataset_list = list(dataset)
        dataset_list = dataset_list
        self.queries = [obj.test_example.query.split(':', 1)[1].strip() for obj in dataset_list]
        self.dataset = iter(dataset_list)
        self.answers = [obj.test_example.answer for obj in dataset_list]
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x
        self.full_dataset = list(dataset_list)
        self._dataset_name = 'prontoqa'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):
        if shuffle_prompt:
            ret = random.sample(list(self.init_prompt), k=num_shot)
        else:
            ret = self.init_prompt[:num_shot]

        if self.sample_prompt_type == "rap":
            return ret

        elif self.sample_prompt_type == "cot":
            # cot or tot
            return get_cot_prompt(ret)
        else:
            raise NotImplementedError

    def eval_output(self, answer, output):
        if output is None:
            return False
        try:
            output = str(output)
            answer = str(answer)
            return output == answer
        except ValueError:
            pass
        try:
            output = str(output)
            answer = str(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer
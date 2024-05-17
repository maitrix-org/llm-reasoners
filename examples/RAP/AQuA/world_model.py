import io
from pyexpat import model
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils
from reasoners.base import Example
import re
import json
import numpy as np


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float
    answer_list: list[str] = None
    answer_values: list[str] = None


MATHState = list[SubResult]
MATHAction = str


class MATHPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


class MATHWorldModel(WorldModel):
    """
    MATH World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model: LanguageModel,
                 n_confidence=8,
                 batch_size=2,
                 temperature=0.8,
                 early_stop_base=None,
                 early_stop_threshold=1.,
                 score_prompts = "/home/xinyuan/workspace/llm-reasoners/examples/AQuA_rap/prompts/score_examples.json"
                 ) -> None:
        super().__init__()
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold
        self.prompt_examples = ""
        self.n_shots = 0
        with open(score_prompts) as f:
            self.score_prompts = json.load(f)

    def update_example(self, example: Example, prompt: MATHPromptDict = None) -> None:
        super().update_example(example, prompt)
        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()
    
    def init_state(self) -> list:
        return []

    def step(self, state: MATHState, action: MATHAction) -> tuple[MATHState, dict]:
        print("********* world model step *******")
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")
            for idx, (q, a, *_) in enumerate(state):
                f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1) + " " + action + "\n")
            f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))
            model_input = f.getvalue()
        
        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        score_dict = defaultdict(list)
        result = ""
        result_count = 0
        answer_count = 0
        none_count = 0
        
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate([model_input] * num,
                                                   hide_input=True,
                                                   do_sample=True,
                                                   temperature=self.temperature,
                                                   eos_token_id='\n').text
                for output in outputs:
                    result = output.strip()
                    result_count += 1
                    if "Now we can" in action:
                        answer = utils.retrieve_answer(result)
                    else:
                        answer = utils.retrieve_answer_not_option(result)
                        
                    if answer is not None:
                        if len(score_dict[(answer, result)])<len(self.score_prompts):
                            for score_prompt_index in range(len(self.score_prompts)):                            
                                with io.StringIO() as f:
                                    f.write(self.score_prompts[score_prompt_index]["input"]+"\n\n")
                                    f.write(self.score_prompts[score_prompt_index]["question_prefix"] + self.example + "\n")
                                    for idx, (q, a, *_) in enumerate(state):
                                        f.write(self.score_prompts[score_prompt_index]["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                                        f.write(self.score_prompts[score_prompt_index]["subanswer_prefix"].format(idx + 1) + " " +  a + "\n")
                                    f.write(self.score_prompts[score_prompt_index]["subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
                                    f.write(self.score_prompts[score_prompt_index]["new_subanswer_prefix"].format(len(state) + 1) + " " + result + "\n")
                                    f.write(self.score_prompts[score_prompt_index]["score_prefix"])
                                    score_input = f.getvalue()
                                    
                                print(f"score_input: {score_input}")
                                
                                logits = self.base_model.get_next_token_logits(score_input, ["Yes", "No"])[0]
                                probs = np.exp(logits) / np.sum(np.exp(logits))
                                score = probs[0]
                                score_dict[(answer, result)].append(score)
                                print(f"score:\n{score}\n")
                        
                        print(f"model output: \n{result}")
                        print(f"retrieved answer: \n{answer}")
                        answer_dict[answer].append(result)
                        print(f"answer {answer_count}: \n{answer}")
                        answer_count += 1
                    else:
                        none_count += 1
                        
                    print("------------------------------")
                    

            # Early stop if confidence is high enough
            '''if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 4 and max_len == len(sorted_answer_dict[1][1]): # change from 2 to 4
                    pass  # Tie with the second best answer
                else:
                    break'''

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            print("Output:", result)
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            result_dict = defaultdict(float)

            for answer_tuple in score_dict:
                result_dict[answer_tuple] = (np.mean(score_dict[answer_tuple]) + len(answer_dict[answer_tuple[0]]) / answer_count) / 2 # test only divide confidence
            
            sorted_answer_dict = sorted(result_dict.items(), key=lambda p: p[1], reverse=True)
            for answer_tuple in sorted_answer_dict:
                print(answer_tuple)
            max_answer = sorted_answer_dict[0]
            answer = max_answer[0][1]  # Here we simply choose the first appearance of the answer
            confidence = max_answer[1]
        
        state.append(SubResult(action, answer, confidence, list(answer_dict.keys()), list(answer_dict.values())))
        print(f"action: \n{action}\nanswer:\n{answer}\nconfidence:{confidence}\n")
        aux = {'confidence': confidence}
        print("********************************")
        return state, aux

    def is_terminal(self, state: MATHState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False
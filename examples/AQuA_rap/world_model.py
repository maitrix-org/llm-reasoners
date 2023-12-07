import io
from pyexpat import model
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils
from reasoners.base import Example
import re
import json

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


class MATHWorldModel(WorldModel[MATHState, MATHAction]):
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
    """
    Question 4: If q is the square of a positive integer, which of the following must be equal to the square of the next positive integer? Options: A) √n + 1, B) n + 1, C) n^2 + 1, D) q + 2√q + 1, E)n^2 + 2n + 1.
    Question 4.1: How to represent the positive integer using q?\nAnswer 4.1: The positive integer is the square root of q. The answer is √q.
    Question 4.2: What is the next positive integer\nAnswer 4.2: The next integer is the positive integer plus 1. The answer is √q+1.\nQuestion {idx}.3: Now we can answer the question with an option from A to E: What is the square of the next integer?\nAnswer {idx}.3: The square of the next integer is (√q+1)^2=q + 2√q + 1. The answer is D.
    {model_input} {subanswer}\n
    The last subquestion is: {subquestion}
    The last subanswer is: {subanswer}
    Question: Is the sub-answer logically correct?
    Answer: 
    """

    def build_score_input(self, model_input, subquestion, subanswer, retrieved_answer):
        score_rules_prompt = """
Question 4: If q is the square of a positive integer, which of the following must be equal to the square of the next positive integer? Options: A) √n + 1, B) n + 1, C) n^2 + 1, D) q + 2√q + 1, E)n^2 + 2n + 1.
Question 4.1: How to represent the positive integer using q?\nAnswer 4.1: The positive integer is the square root of q. The answer is √q.
Question 4.2: What is the next positive integer\nAnswer 4.2: The next integer is the positive integer plus 1. The answer is √q+1.
Is the sub-answer logically correct? Yes. 

{model_input} {subanswer}\n
The last subquestion is: {subquestion}
The last subanswer is: {subanswer}
Question: Is the sub-answer logically correct?
Answer:
""".strip()

        return score_rules_prompt.format(model_input=model_input.strip(),
                                         subquestion=subquestion.strip(),
                                         subanswer=subanswer.strip(),
                                         retrieved_answer=retrieved_answer.strip())
        
    def cal_score(self, state: MATHState, action: MATHAction, ):
        yes_matches = re.findall(r'\byes\b', score_output, re.IGNORECASE)
        no_matches = re.findall(r'\bno\b', score_output, re.IGNORECASE)

        yes_count = len(yes_matches)
        no_count = len(no_matches)
        return {yes_count, no_count}
    
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
        result = ""
        result_count = 0
        answer_count = 0
        
        
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
                    print(f"action: \n{action}")
                    result = output.strip()
                    print(f"result {result_count}: \n{result}")
                    result_count += 1
                    if "Now we can" in action:
                        answer = utils.retrieve_answer(result)
                    else:
                        answer = utils.retrieve_answer_not_option(result)
                        
                    if answer is not None:
                        with io.StringIO() as f:
                            f.write(self.useful_prompt["input"])
                            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
                            for idx, (q, a, *_) in enumerate(state):
                                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
                            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
                            f.write(self.useful_prompt["useful_prefix"])
                            model_input = f.getvalue()
                        score_input = self.build_score_input(
                            model_input=model_input,
                            subquestion=action,
                            subanswer=output,
                            retrieved_answer=answer)
                        print(f"score_input:\n{score_input}")
                        score_output = self.base_model.generate(
                            [score_input],
                            hide_input=True,
                            do_sample=True,
                            temperature=0,
                            eos_token_id='\n').text
                        print(f"score_output:\n{score_output}")
                        score = self.cal_score(score_output=score_output[0])
                        
                    if answer is not None:
                        print(f"model output: \n{result}")
                        print(f"retrieved answer: \n{answer}")
                        answer_dict[answer].append(result)
                        print(f"answer {answer_count}: \n{answer}")
                        answer_count += 1
                        
                        
                    print("------------------------------")
                    

            # Early stop if confidence is high enough
            if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 2 and max_len == len(sorted_answer_dict[1][1]):
                    pass  # Tie with the second best answer
                else:
                    break

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            print("Output:", result)
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())
        print(answer_dict.keys())
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
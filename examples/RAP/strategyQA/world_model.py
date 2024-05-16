import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float


StrategyQAState = list[SubResult]
StrategyQAAction = str


class StrategyQAPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    answer_prefix: str
    overall_question_prefix: str


#class StrategyQAWorldModel(WorldModel[StrategyQAState, StrategyQAAction]):
class StrategyQAWorldModel(WorldModel):
    """
    strategyQA World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_confidence=8,
                 batch_size=2,
                 temperature=0.8,
                 eos_token_id='\n',
                 early_stop_base=None,
                 early_stop_threshold=1.) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt: StrategyQAPrompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold
        self.eos_token_id = eos_token_id

    def init_state(self) -> list:
        return []

    def step(self, state: StrategyQAState, action: StrategyQAAction) -> tuple[StrategyQAState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt["input"])
            f.write(self.prompt["question_prefix"] + self.example + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(self.prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.prompt["answer_prefix"].format(len(state) + 1))
            model_input = f.getvalue()

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        # print(f'====\nsubanswer prompt: {model_input}\n====')
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate([model_input] * num,
                                                   hide_input=True,
                                                   do_sample=True,
                                                   temperature=self.temperature,
                                                   eos_token_id=self.eos_token_id).text
                for output in outputs:
                    result = output.strip()
                    # print(f"subanswer output: {result}")
                    answer = utils.retrieve_answer(result)
                    if answer is not None:
                        answer_dict[answer].append(result)
                    # print(f"subanswer output (extracted): {answer}")

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
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        # print(confidence)
        aux = {'confidence': confidence}
        return state, aux

    def is_terminal(self, state: StrategyQAState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            ## try word match
            last_sub_words = set(state[-1].sub_question.lower().split(' '))
            overall_ques_words = set(self.example.lower().split(' '))
            new_words = last_sub_words - overall_ques_words
            if len(new_words) <= 1:
                return True
        return False

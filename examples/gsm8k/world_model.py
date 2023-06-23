import re
import io
from collections import defaultdict
from rap import WorldModel


class GSMWorldModel(WorldModel[list[tuple[str, str, float]], str]):
    """ GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model,
                 prompt,
                 n_confidence=8,
                 batch_size=2,
                 early_stop_with_confidence=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        # Confidence threshold for early stop ; confidence cannot exceed 1.0, so 1.1 means no early stop
        self.early_stop_conf = early_stop_with_confidence if early_stop_with_confidence is not None else 1.1

    def init_state(self) -> list:
        return []

    def step(self, state: list, action: str) -> list:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt["input"])
            f.write(self.prompt["question_prefix"])
            f.write(self.example + "\n")
            for idx, (q, a, c) in enumerate(state):
                f.write(self.prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.prompt["answer_prefix"].format(len(state) + 1))
            model_input = f.getvalue()

        answer_dict = defaultdict(list) # map from answer to list of thoughts
        for idx in range(0, self.n_confidence, self.batch_size):
            n_samples = min(self.n_confidence - idx, self.batch_size)
            outputs = self.base_model([model_input] * n_samples, end_token="\n", hide_input=True)["text"]
            for output in outputs:
                result = output.strip()
                match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
                if match is None:
                    continue
                sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
                answer_dict[sub_answer].append(output)
                answer_list.append(sub_answer)
            if len(answer_dict) == 0:
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len < 2:
                continue
            if len(sorted_answer_dict) < 2:
                break
            second_max_len = len(sorted_answer_dict[1][1])
            if max_len >= len(answer_dict) / 2 and max_len > second_max_len:
                break

        if len(answer_dict) == 0:
            confidence, answer = 0, ""
        else:
            answer = sorted_answer_dict[0][1][0]  # [0]: maximum; [1]: list of outputs; [0]: first output in the list
            confidence = max_len / len(answer_list)

        state.append((action, answer, confidence))
        return state

    def is_terminal(self, state: list) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1][0]:
            return True
        else:
            return False

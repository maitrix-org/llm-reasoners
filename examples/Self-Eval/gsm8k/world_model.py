from reasoners import WorldModel, LanguageModel
from typing import NamedTuple
from prompt import choice_prefix, evaluate_prompt
from utils import get_indent
import io
import regex
import math
from reasoners.lm import OpenAIModel, Llama2Model, Llama3Model

class SubResult(NamedTuple):
    action: str
    action_prob: float
    action_length: int
    evaluation: str
    action_confidence: float

GSM8kState = list[SubResult]
# action should include both the action and the action_prob
GSM8kAction = (str, float, int) # (action, action_prob, action_length)
GSM8kExample = str


class GSM8kWorldModel(WorldModel[GSM8kState, GSM8kAction, GSM8kExample]):
    """
    GSM8k World Model
    State: [(action, action_prob, evaluation, action_confidence), ...]
    Action: step
    """

    def __init__(self,
                base_model: LanguageModel,
                eval_num: int = 1, # only 1 evaluation is needed
                temperature: float = 0.0 # set to greedy for confidence
                ) -> None:
        
        super().__init__()

        self.base_model = base_model
        self.eval_num = eval_num
        self.temperature = temperature

    def init_state(self) -> list:
        return []

    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(evaluate_prompt)
            f.write("\n\n\n\n\n")
            f.write(f'Q: {self.example}\n\n# solution in Python:\n\n\ndef solution():\n    """{self.example}"""\n')
            for a, _, _, e, _ in state:
                f.write(f"{a}\n")
                # get the indent of action
                indent = get_indent(a)
                # add indent to every choice prefix (list)
                f.write("\n".join([indent + prefix for prefix in choice_prefix]))
                f.write(f"{e}\n")
            
            f.write(f"{action[0]}\n")
            # get the indent of action
            indent = get_indent(action[0])
            # add indent to every choice prefix (list)
            f.write("\n".join([indent + prefix for prefix in choice_prefix]))

            model_input = f.getvalue()

        if isinstance(self.base_model, OpenAIModel):
            eos_token_id = []
        elif isinstance(self.base_model, Llama2Model):
            eos_token_id = ["\n"]
        elif isinstance(self.base_model, Llama3Model):
            eos_token_id = ["\n\n", ".\n", ".\n\n","\n"]
            
        outputs = self.base_model.generate(model_input,
                                    temperature=self.temperature,
                                    max_tokens=64,
                                    top_p=1,
                                    num_return_sequences=self.eval_num,
                                    stop="\n",
                                    logprobs=5,
                                    hide_input=True,
                                    do_sample=True,
                                    eos_token_id=eos_token_id)

        # outputs = self.base_model.generate(prompt=model_input,
        #                                     max_tokens=64,
        #                                     temperature=self.temperature,
        #                                     top_p=1,
        #                                     num_return_sequences=self.eval_num,
        #                                     stop="\n",
        #                                     logprobs=5)
        
        # https://github.com/YuxiXie/SelfEval-Guided-Decoding/blob/6bb8c0b2297d41e86e12a0e2ec4b49818b577420/src/utils/tool.py#L311
        def _check_eq(x, tokens):
            x = regex.sub(r'[\(\)\s]', ' ', x).strip()
            eq = False
            if any(x == t for t in tokens): 
                eq = True
            elif any(x.lower() == t.lower() for t in tokens): 
                eq = True
            return eq
        
        action_confidence_list, evaluation_list = [], []

        for i in range(self.eval_num):
            text = outputs.text[i]
            tokens = outputs.log_prob[i]["tokens"]
            top_logprobs = outputs.log_prob[i]["top_logprobs"]

            right_tokens, wrong_tokens = ['A'], ['B']

            action_confidence = 0
            for t, tlp in zip(tokens, top_logprobs):
                # if there is no 'A' or 'B' in the tokens, then skip
                if not _check_eq(t, wrong_tokens + right_tokens) or t not in text: 
                    continue
                # convert log prob to prob
                tlp = {k: math.exp(v) for k,v in tlp.items()}
                # get the confidence of the action
                action_confidence = sum(tlp.get(k, 0) for k in tlp if _check_eq(k, right_tokens))
                # only the first token is considered
                break

            action_confidence_list.append(action_confidence)
            evaluation_list.append(text)
        
        # get the evaluation with the highest confidence
        action_confidence = max(action_confidence_list)
        evaluation = evaluation_list[action_confidence_list.index(action_confidence)]

        state.append(SubResult(
            action=action[0], action_prob=action[1], action_length=action[2],
            evaluation=evaluation, action_confidence=action_confidence))

        return state, {"action_confidence": action_confidence}

    def is_terminal(self, state: GSM8kState) -> bool:
        # several ways to terminate:
        # 1. state > 0
        # 2. the last state's action is "    return result" (do rstrip() to remove \n)
        # 3. or the last state is empty 

        if len(state) > 0:
            
            if "return result" in state[-1].action or state[-1].action == "":
                return True
        
        return False
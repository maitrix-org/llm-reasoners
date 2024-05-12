import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import StrategyQAState, StrategyQAAction, StrategyQAPrompt
from reasoners import SearchConfig, LanguageModel
import utils


class strategyQAUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str


class StrategyQAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 useful_prompt: dict,
                 decompose_prompt: str,
                 n_actions=4,
                 batch_size=2,
                 temperature=0.8,
                 eos_token_id='\n',
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = ''
        self.prompt: StrategyQAPrompt = prompt
        self.useful_prompt: strategyQAUsefulPrompt = useful_prompt
        self.decompose_prompt = decompose_prompt
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None
        self.subquestion_conf = {'Yes': 1.0, 'Maybe':0.5, 'No':0.1}

    def update_example(self, example: str, prompt = None) -> None:
        super().update_example(example)
        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example)[1]
            self.overall_question = self.example

    def get_actions(self, state: StrategyQAState, ) -> list[StrategyQAAction]:
        with io.StringIO() as f:
            if len(state) == 0:
                f.write(self.decompose_prompt + '\n\nQ: ' + self.overall_question + '\nA: To answer the question \"' + self.overall_question + '\", we need to know:')
            else:
                f.write(self.prompt["input"])
                f.write(self.prompt["question_prefix"] + self.example + "\n")
                for idx, (q, a, _) in enumerate(state):
                    f.write(self.prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                    f.write(self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
                f.write(self.prompt["subquestion_prefix"].format(len(state) + 1))
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])

            model_input = f.getvalue()
        # n_actions = 1 if at_depth_limit else self.n_actions
        # temperature = 0 if at_depth_limit else self.temperature
        n_actions = self.n_actions
        temperature = self.temperature
        outputs = []
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                eos_token_id=self.eos_token_id).text
        # print(f'====\nsub-question prompt: {model_input}\n====')
        # print(f"====\nsub-question: {outputs}\n====")
        outputs = [output.strip() for output in outputs]
        if len(state) == 0:
            for i, output in enumerate(outputs):
                # print(f"sub-question output: {output}")
                subqs_list = utils.extract_subquestions(output[:-1])
                print('\n<<<< sub-questions list >>>>\n{}'.format(subqs_list))
                q1 = subqs_list[0]
                if q1[0] != '"':
                    q1 = '"' + q1
                if q1[-1] != '"':
                    q1 = q1 + '"'
                # print('\n<<<< Q1 >>>>\n{}'.format(subq_format))
                outputs[i] = q1[1:-1]
        # print(f"====\nsub-question: {outputs}\n====")
        ### similar to is_terminal function in world
        if at_depth_limit:
            outputs = [self.prompt["overall_question_prefix"] + ' ' + self.overall_question]
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.prompt["overall_question_prefix"] in output:
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                last_sub_words = set(output.lower().split(' '))
                overall_ques_words = set(self.overall_question.lower().split(' '))
                new_words = last_sub_words - overall_ques_words
                if len(new_words) <= 1:
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        # print(f"====\nsub-question output after process: {outputs}\n====")

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def fast_reward(self, state: StrategyQAState, action: StrategyQAAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue().replace('Now we can answer the question: ', '')

        # print(f'====\nreward input: {model_input}====\n')
        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)
        # print(f'original prob: {probs}, r_useful: {useful_prob}, fast_reward: {fast_reward}')
        return fast_reward, {'r_useful': useful_prob}
    
    # def fast_reward(self, state: StrategyQAState, action: StrategyQAAction) -> tuple[float, dict]:
    #     with io.StringIO() as f:
    #         f.write(self.useful_prompt["input"])
    #         f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
    #         for idx, (q, a, _) in enumerate(state):
    #             f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
    #             f.write(self.useful_prompt["subanswer_prefix"].format(idx + 1) + " " + a + "\n")
    #         f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
    #         f.write(self.useful_prompt["useful_prefix"])
    #         model_input = f.getvalue().replace('Now we can answer the question: ', '')

    #     # print(f'====\nreward input: {model_input}')
    #     outputs = []
    #     for idx in range(0, self.n_actions, self.batch_size):
    #         n_samples = min(self.n_actions - idx, self.batch_size)
    #         outputs += self.base_model.generate([model_input] * n_samples,
    #                                             hide_input=True,
    #                                             do_sample=True,
    #                                             temperature=self.temperature,
    #                                             eos_token_id='\n\n').text
    #     # print(f"sub-question reward: {outputs}")
    #     outputs = [output.strip().split('.')[0] for output in outputs]
    #     # print(f"processed reward: {outputs}\n====")
    #     useful_prob =  sum(self.subquestion_conf[output] if output in self.subquestion_conf else 0 for output in outputs)
    #     fast_reward, _ = self.calculate_reward(useful_prob)
    #     # print(f'r_useful: {useful_prob}, fast_reward: {fast_reward}')
    #     return fast_reward, {'r_useful': useful_prob}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: StrategyQAState, action: StrategyQAAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)

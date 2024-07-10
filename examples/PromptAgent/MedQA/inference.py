import os
import re
import json
import random
from typing import NamedTuple, List, Tuple, Dict, Any
from config import *
from reasoners import WorldModel, LanguageModel,Reasoner,SearchConfig
from reasoners.algorithm import MCTS
import fire
from task import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class PromptState(NamedTuple):
    text: str
    trajectory: List[str]
    accuracy_trajectory: List[float]
    depth: int

class PromptAction(NamedTuple):
    new_prompt: str


class PromptWorldModel(WorldModel[PromptState, PromptAction, str]):
    def __init__(self, language_model: LanguageModel, eval_data,test_data, depth_limit: int = 2, origin_prompt="Answer the following question.") -> None:
        super().__init__()
        self.eval_data = eval_data
        self.test_data = test_data
        self.language_model = language_model
        self.origin_prompt = origin_prompt
        self.accuracy_cache = {}
        self.example = None
        self.best_accuracy= 0
        self.depth_limit = depth_limit

    def update_example(self, example: str, prompt: dict = None) -> None:
        self.example = example

    def init_state(self) -> PromptState:
        return PromptState(self.origin_prompt, [], [], 0)

    def step(self, state: PromptState, action: PromptAction) -> Tuple[PromptState, dict]:
        new_text = action.new_prompt
        new_trajectory = state.trajectory + [state.text]
        temp_accuracy=self.get_accuracy(state.text)
        new_accuracy_trajectory = state.accuracy_trajectory + [temp_accuracy]
        if temp_accuracy>self.best_accuracy:
            self.best_accuracy=temp_accuracy
            print("New Best Prompt: ",state.text)
            print("The Evaluate Accuracy: ",temp_accuracy)
            print("________________________________________________")
        new_state = PromptState(new_text, new_trajectory, new_accuracy_trajectory, state.depth + 1)
        return new_state, {}

    def is_terminal(self, state: PromptState) -> bool:
        if state.depth >= self.depth_limit:
            return True
        if state.depth > 2:
            parent_reward = state.accuracy_trajectory[-1] if state.accuracy_trajectory else 0
            root_reward = state.accuracy_trajectory[0] if state.accuracy_trajectory else 0
            min_threshold = (parent_reward + root_reward) / 2
            max_threshold = self.best_accuracy
            current_reward = self.get_accuracy(state.text)
            if current_reward < min_threshold or current_reward > max_threshold:
                if current_reward > self.best_accuracy:
                    self.best_accuracy = current_reward
                    print("New Best Prompt: ",state.text)
                    print("The Evaluate Accuracy: ",current_reward)
                    print("________________________________________________")
                return True
        return False

    def get_accuracy(self, prompt: str,test_mode=False) -> float:
        if not test_mode:
            if prompt in self.accuracy_cache:
                return self.accuracy_cache[prompt]
            questions = self.eval_data
            correct = 0
            for question in questions:
                if prompt_position=="pre":
                    inputs = f"{prompt}\nQuestion: {question['question']}. At the end show the answer option between <answer> and </answer>.\n"
                elif prompt_position=="pos":
                    inputs = f"Question: {question['question']}.\n{prompt} At the end show the answer option between <answer> and </answer>.\n"
                else:
                    print("invalid prompt position")
                outputs = self.language_model.generate(prompt=[inputs])
                answer = extract_answer(outputs.text[0])
                if check_anwser(answer, question["answer"]):
                    correct+=1
            accuracy = correct / len(questions)
            self.accuracy_cache[prompt] = accuracy
            return accuracy
        else:
            questions = self.test_data
            correct = 0
            for question in questions:
                if prompt_position=="pre":
                    inputs = f"{prompt}\nQuestion: {question['question']}. At the end show the answer option between <answer> and </answer>.\n"
                elif prompt_position=="pos":
                    inputs = f"Question: {question['question']}.\n{prompt} At the end show the answer option between <answer> and </answer>.\n"
                else:
                    print("invalid prompt position")
                outputs = self.language_model.generate(prompt=[inputs])
                answer = extract_answer(outputs.text[0])
                if check_anwser(answer, question["answer"]):
                    correct+=1
            accuracy = correct / len(questions)
            self.accuracy_cache[prompt] = accuracy
            return accuracy

class PromptSearchConfig(SearchConfig[PromptState, PromptAction, str]):
    def __init__(self, world_model: PromptWorldModel,lm_model: LanguageModel , optimize_model : LanguageModel, num_batches: int = 3, steps_per_gradient: int = 1,batch_size: int=5) -> None:
        super().__init__()
        self.world_model = world_model
        self.example = None
        self.lm_model = lm_model
        self.num_batches = num_batches
        self.steps_per_gradient = steps_per_gradient
        self.batch_size = batch_size
        self.optimize_model=optimize_model
        
    def update_example(self, example, prompt: dict = None) -> None:
        self.example = example

    def get_actions(self, state: PromptState) -> List[PromptAction]:
        
        actions = []
        questions = self.example
        batch_index = 0
        while batch_index < self.num_batches:
            error_strings = []
            new_prompts=[]
            sample_questions = random.sample(questions, self.batch_size)
            if prompt_position=="pre":
                prompt_with_questions = [
                f"{state.text}\nQuestion: {question['question']}. At the end show the answer option between <answer> and </answer>.\n"
                for question in sample_questions
                ]
            elif prompt_position=="pos":
                prompt_with_questions = [
                f"Question: {question['question']}\n {state.text}\n. At the end show the answer option between <answer> and </answer>.\n"
                for question in sample_questions
                ]
            else:
                    print("invalid prompt position")
            prompt_with_questions = [
                f"{state.text}\nQuestion: {question['question']}. At the end show the answer option between <answer> and </answer>.\n"
                for question in sample_questions
            ]
            generated_texts=[]
            for prompt_with_question in prompt_with_questions:
                inputs = prompt_with_question
                outputs = self.lm_model.generate(prompt=[inputs])
                generated_texts.append(outputs.text[0])
            
            has_errors = False
            ind=0
            for _, (generated_text, sample_question) in enumerate(zip(generated_texts,sample_questions)):
                answer = extract_answer(generated_text)
                if not check_anwser(answer,sample_question['answer']):
                    has_errors = True
                    ind+=1
                    error_string = f"""
                                error string <{ind}>
                                The model's input is: {f"{state.text} Question: {sample_question['question']}"}
                                The model's response is: {generated_text}
                                The correct label is: {sample_question['answer']}
                                The model's prediction is: {answer}
                                """
                    error_strings.append(error_string)
            if not has_errors:
                continue

            if error_strings:
                trajectory_prompts = "\n".join(state.trajectory)
                error_feedback_prompt = f"""
                                        I'm writing prompts for a language model designed for a task.
                                        My current prompt is: {state.text}
                                        But this prompt gets the following examples wrong:
                                        {"".join(error_strings)}
                                        For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
                                         """
                error_feedback_output = self.optimize_model.generate(prompt=[error_feedback_prompt])
                error_feedback_text = error_feedback_output.text[0]
                state_transit_prompt = f"""
                                        I'm writing prompts for a language model designed for a task.
                                        My current prompt is: {state.text}
                                        But this prompt gets the following examples wrong:
                                        {"".join(error_strings)}
                                        Based on these errors, the problems with this prompt and the reasons are:
                                        {error_feedback_text}
                                        There is a list of former prompts including the current prompt, and each prompt is modified from its former prompts:
                                        {trajectory_prompts}
                                        Based on the above information, please write {self.steps_per_gradient} new prompts following these guidelines:
                                        1. The new prompts should solve the current prompt's problems.
                                        2. The new prompts should consider the list of prompts and evolve based on the current prompt.
                                        3. Each new prompt should be wrapped with <START> and <END>.
                                        The new prompts are:
                                        """
                
                new_prompts_output = self.optimize_model.generate(prompt=[state_transit_prompt])
                new_prompts_text = new_prompts_output.text[0]
                new_prompts = re.findall(r'<START>(.*?)<END>', new_prompts_text, re.DOTALL)
                if len(new_prompts)==0:
                    continue
                for new_prompt in new_prompts:
                    actions.append(PromptAction(new_prompt.strip()))
                batch_index += 1
        print(actions)
        print("***************************************")
        return actions

    def reward(self, state: PromptState, action: PromptAction) -> Tuple[float, dict]:
        return self.world_model.get_accuracy(action.new_prompt), {}
    
def optimize_prompt(train_data, questions_eval, questions_test):
    # Initialize models
    
    # Initialize the world model
    world_model = PromptWorldModel(base_model,eval_data=questions_eval, test_data=questions_test, depth_limit=depth_limit,origin_prompt=origin_prompt)

    # Configure search parameters
    search_config = PromptSearchConfig(
        world_model=world_model, 
        lm_model=base_model, 
        optimize_model=optimize_model, 
        num_batches=num_batches, 
        steps_per_gradient=steps_per_gradient, 
        batch_size=batch_size
    )

    # Initialize the search algorithm
    search_algo = MCTS(
        output_trace_in_each_iter=True, 
        w_exp=w_exp, 
        n_iters=n_iters, 
        depth_limit=depth_limit
    )

    # Initialize the reasoner
    reasoner = Reasoner(
        world_model=world_model, 
        search_config=search_config, 
        search_algo=search_algo
    )
    questions=train_data
    optimized_result = reasoner(questions)
    best_prompt = optimized_result.trace_of_nodes[0].state.text
    best_accuracy = optimized_result.trace_of_nodes[1].state.accuracy_trajectory[0]
    with open("result.txt", "w") as file:
        file.write(f"Original Prompt: {origin_prompt}\n")
        file.write(f"Original Eval Data Accuracy: {best_accuracy}\n")
    for node in optimized_result.trace_of_nodes:
        if node.reward > best_accuracy:
            best_accuracy = node.reward
            best_prompt = node.state.text
    with open("result.txt", "a") as file:
        file.write(f"Best Prompt: {best_prompt}\n")
        file.write(f"Best Eval Data Accuracy: {best_accuracy}\n")
        file.write(f"Original Test Data Accuracy: {world_model.get_accuracy(origin_prompt,test_mode=True)}\n")
        file.write(f"Best Test Data Accuracy: {world_model.get_accuracy(best_prompt,test_mode=True)}")
    
    


def main():
    questions_train,questions_eval,questions_test = load_task_dataset()
    questions_train = reformat_data(questions_train)
    questions_eval = reformat_data(questions_eval)
    questions_test= reformat_data(questions_test)
    optimize_prompt(questions_train, questions_eval,questions_test)

if __name__ == "__main__":
    fire.Fire(main)
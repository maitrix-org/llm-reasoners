from reasoners import LanguageModel, SearchConfig, GenerateOutput
import random
import os
from datetime import datetime

class OrderQAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 check_endstate_prompt: str,  
                 check_answer_prompt: str,  
                 fast_reward_prompt: str,     
                 reward_prompt: str, 
                 check_answer_available_prompt: str,
                 extract_answer_prompt: str,
                 QApairs: dict,       
                 reward_alpha=0.5,
                 goal_reward_default=0.0,
                 goal_reached_reward=1.0,
                 trace_number=0) -> None:
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.fast_reward_prompt = fast_reward_prompt
        self.check_answer_prompt = check_answer_prompt
        self.reward_prompt = reward_prompt
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.check_endstate_prompt = check_endstate_prompt
        self.check_answer_available_prompt = check_answer_available_prompt
        self.extract_answer_prompt = extract_answer_prompt
        self.trace_number=trace_number
        self.QApairs=QApairs

    # FRQ isEnd
    def isEnd(self, state):
        if len(state) == 0:
            return 0
        
        # Construct prompt
        prompt_question = 'Now, the problem statement is:\n' + self.prompt
        reasoning_trace = ''.join(f'\n{state[i]}' for i in range(len(state)))
        prompt_trace = 'The reasoning steps so far are:' + reasoning_trace
        prompt_final = self.check_endstate_prompt + '\n\n' + prompt_question + '\n\n' + prompt_trace + '\n\n' + "Now, identify if the last line of the reasoning gives out the final answer. Your output should ONLY be \"0\" or \"1\". You should first consider the speical case."
        
        # print("isEnd final prompt:" + prompt_final)
        
        output = ""
        cnt = 0
        while output not in ["0", "1"]:
            model_output : GenerateOutput = self.base_model.generate(prompts=[prompt_final], temperature=0)
            output = model_output.text[0]
            cnt+=1
            if cnt > 5:
                return -1
        return int(output)
    
    def isAnswerCorrect(self, answer, correct_answer, question):
        prompt_tail = self.check_answer_prompt
        # prompt_question = 'The problem statement is:\n' + question
        prompt_answer = 'The student\'s answer is:\n' + answer
        prompt_correct_answer = 'The official correct answer is:\n' + correct_answer
        prompt_final = prompt_tail + '\n\n' + prompt_answer + '\n\n' + prompt_correct_answer
        output = ""
        cnt = 0
        while output not in ["0", "1"]:
            model_output : GenerateOutput = self.base_model.generate(temperature=0, prompts=[prompt_final])
            output = model_output.text[0]
            # print(f"isAnswerCorrectTrial: {output}")
            cnt+=1
            if cnt > 5:
                return -1
        # print(f"isAnswerCorrect: {int(output)}")
        return int(output)
    
    def isAnswerAvailable(self, state):
        if len(state) == 0:
            return 0
        
        # Construct prompt
        prompt_question = 'Now, the problem statement is:\n' + self.prompt
        reasoning_trace = ''.join(f'\n{state[i]}' for i in range(len(state)))
        prompt_trace = 'The reasoning steps so far are:' + reasoning_trace
        prompt_final = self.check_answer_available_prompt + '\n\n' + prompt_question + '\n\n' + prompt_trace + '\n\n' + "Now, judge if the final answer can be extracted from the reasoning trace so far. Your output should ONLY be \"0\" or \"1\"."
        
        output = ""
        cnt = 0
        while output not in ["0", "1"]:
            model_output : GenerateOutput = self.base_model.generate(prompts=[prompt_final], temperature=0)
            output = model_output.text[0]
            cnt+=1
            if cnt > 5:
                return -1
        return int(output)
    
    def extract_answer(self, state):
        # Construct prompt
        prompt_question = 'Now, the problem statement is:\n' + self.prompt
        reasoning_trace = ''.join(f'\n{state[i]}' for i in range(len(state)))
        prompt_trace = 'The reasoning steps so far are:' + reasoning_trace
        prompt_final = self.extract_answer_prompt + '\n\n' + prompt_question + '\n\n' + prompt_trace
        
        model_output : GenerateOutput = self.base_model.generate(prompts=[prompt_final], temperature=0)
        
        if self.trace_number != 0:
            os.makedirs(f"step_traces{self.trace_number}", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"step_traces{self.trace_number}/step_{timestamp}.txt"
            with open(filename, "w") as file:
                file.write(prompt_final)
                file.write("\n\nNext Possible Reasoning Steps:\n")
                for output in model_output.text:
                    file.write(f"{output}\n")

        return model_output.text
    
    def get_actions(self, state: list) -> list:
        if self.isAnswerAvailable(state):
            return self.extract_answer(state)
            
        # Construct prompt
        prompt_question = 'Now, the problem statement is:\n' + self.prompt
        reasoning_trace = ''.join(f'\n{state[i]}' for i in range(len(state)))
        if len(state) == 0:
            prompt_trace = 'There is no reasoning step now. Please provide the first line of reasoning.'
        else:
            prompt_trace = 'The reasoning steps so far are:' + reasoning_trace
        prompt_final = self.example + '\n\n' + prompt_question + '\n\n' + prompt_trace
        
        # Generate the output using the model
        prompts = [prompt_final]
        model_output: GenerateOutput = self.base_model.generate(temperature=1, prompts=prompts, attempts=2)

        if self.trace_number != 0:
            os.makedirs(f"step_traces{self.trace_number}", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"step_traces{self.trace_number}/step_{timestamp}.txt"
            with open(filename, "w") as file:
                file.write(prompt_final)
                file.write("\n\nNext Possible Reasoning Steps:\n")
                for output in model_output.text:
                    file.write(f"{output}\n")

        return model_output.text

    # random reward
    # def fast_reward(self, state, action):
    #     return random.random(), {}

    # def reward(self, state, action, **kwargs):
    #     return random.random(), {}
    
    def fast_reward(self, state, action):
        # if(len(state)==0):
            # print("No reasoning trace yet. Heed the fast_reward now:")
        if self.isEnd(state + [action]) :
            # print(f"Reached a final answer. The answer is {action}. fast_reward set to 2.0")
            return 2.0, {"relevance_score": 2.0}
            
        
        prompt = self.fast_reward_prompt
        question = self.prompt
        reasoning = "\n".join(state)
        next_step = action
        final_prompt = prompt + "\n\nThe problem statement is:\n" + question + "\n\nThe incomplete reasoning trace is:\n" + reasoning + "\n\n\nNow, decide if this additional line of reasoning is \"positive\", \"neutral\", or \"negative\" to be the next line of reasoning after the incomplete reasoning trace in the context of solving the original problem:\n" + next_step + "\n\nOn the last line of your output, you must ONLY put the word \"positive\", \"neutral\" or \"negative\" itself."
        
        # print(final_prompt)
        candidates = ["positive", "neutral", "negative"]
        weights = {"positive": 2, "neutral": 1, "negative": -2}

        llm_eval_result = self.base_model.get_specific_answer_in_last_line(prompt=final_prompt, candidates=candidates, attempts=10)
        
        total = sum(llm_eval_result.values())

        probabilities = {key: count / total for key, count in llm_eval_result.items()}
        
        relevence_score = sum(probabilities[candidate] * weights[candidate] for candidate in llm_eval_result)
        # print(f"fast_reward: {relevence_score}")
        
        reward_value = max(0.1, relevence_score)
        return reward_value, {"relevance_score": relevence_score}

    # reward without correct answer
    def reward(self, state, action, **kwargs) -> tuple:
        relevance_score = kwargs.get("relevance_score", 0)
        
        # Calculate end-state goal reward
        if self.isEnd(state + [action]):
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = self.goal_reward_default

        # Calculate reward as weighted sum of relevance and goal reward
        # reward_value = self.reward_alpha * relevance_score + (1 - self.reward_alpha) * goal_reward
        reward_value = 1.0 + goal_reward
        # print(f"Final reward: {reward_value}")
        return reward_value, {
            "relevance_score": relevance_score,
            "goal_reward": goal_reward
        }
    
    # reward with correct answer
    # def reward(self, state, action, **kwargs) -> tuple:
    #     relevance_score = kwargs.get("relevance_score", 0)
        
    #     # Calculate end-state goal reward
    #     if self.isEnd(state + [action]):
    #         correct_answer = next((item["answer"] for item in self.QApairs if item["question"] == self.prompt), None)
    #         if self.isAnswerCorrect(answer=action, correct_answer=correct_answer, question=self.prompt):
    #             goal_reward = self.goal_reached_reward
    #         else:
    #             goal_reward = self.goal_reached_reward * (-1)
    #     else:
    #         goal_reward = self.goal_reward_default

    #     # Calculate reward as weighted sum of relevance and goal reward
    #     reward_value = self.reward_alpha * relevance_score + (1 - self.reward_alpha) * goal_reward
    #     return reward_value, {
    #         "relevance_score": relevance_score,
    #         "goal_reward": goal_reward
    #     }
    
    # Deprecated logprob based fast reward functions, may break the endpoint
    def logprob_fast_reward(self, state, action) -> tuple:
        # Use a rough evaluation based on relevance and progress
        relevance_score = self.base_model.get_loglikelihood(
            "\n".join(state), [action])[0]
        
        # Slight positive bias to avoid zero reward when action is relevant
        reward_value = max(0.1, relevance_score)
        return reward_value, {"relevance_score": relevance_score}
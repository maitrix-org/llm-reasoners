from reasoners import SearchConfig, LanguageModel as Model
from world_model import PromptAlignWorldModel, PromptAlignState, PromptAlignAction, PromptAlignExample
from prompt import optimize_prompt
import json
import logging
from log_format import prompt_log, output_log, info_log
from utils import parse_json_output

class PromptAlignSearchConfig(SearchConfig[PromptAlignState, PromptAlignAction, PromptAlignExample]):
    def __init__(self,
                 optimize_model: Model,
                 n_actions: int = 10,
                 temperature: float = 0.7
                 ):
        super().__init__()
        self.optimize_model = optimize_model
        self.n_actions = n_actions
        self.temperature = temperature
        
        # logging
        logging.info("PromptAlignSearchConfig initialized with n_actions=%d, temperature=%f", n_actions, temperature)
    
    def get_actions(self, state: PromptAlignState) -> list[PromptAlignAction]:
        # logging
        logging.info("Generating actions for the current state")
        
        # we need current system prompt, current query, current output and current eval_dict
        current_system_prompt = state[-1].system_prompt
        current_query = state[-1].query
        current_output = state[-1].output
        current_eval_dict = state[-1].eval_dict

        if len(current_eval_dict) == 0:
            logging.info(info_log.format(info="Error in output parsing, skipping optimizarion"))
            return True, [current_system_prompt] 

        score = 0
        for aspect in current_eval_dict:
            score += int(current_eval_dict[aspect]["score"])
        score /= len(current_eval_dict)

        
        # first let's check whether all eval_dict scores are 5
        if all([int(current_eval_dict[aspect]["score"]) == 5 for aspect in current_eval_dict]):
            # skip the optimization if all scores are 5
            logging.info(info_log.format(info="All scores are 5, skipping optimization"))
            
            return True, [current_system_prompt] 
        
        elif score > 4.5:
            # skip the optimization if avg scores is > 4.5
            logging.info(info_log.format(info="Avg score is >4.5, skipping optimization."))

            return True, [current_system_prompt] 
        

        # we also need all the previous system prompts
        previous_system_prompts = [sub_result.system_prompt for sub_result in state]
        # but we only need the last 5
        previous_system_prompts = previous_system_prompts[-5:]
        
        # construct the prompt
        prompt = optimize_prompt.replace("[CURRENT_SYSTEM_PROMPT]", current_system_prompt)\
                                .replace("[QUERY]", current_query)\
                                .replace("[OUTPUT]", current_output)\
                                .replace("[OUTPUT_EVALUATION]", json.dumps(current_eval_dict, indent=4))\
                                .replace(
                                    "[FORMER_SYSTEM_PROMPTS]",
                                    "\n".join(f"---Version {i+1}---\n{p}" for i, p in enumerate(previous_system_prompts[:-1])) + "\n---Current Version---\n" + previous_system_prompts[-1]
                                    )
                                
        # logging the prompt, use "debug" level for the prompt
        logging.debug(prompt_log.format(prompt=prompt))
        
        # generate the new system prompt
        outputs = self.optimize_model.generate(
            user_prompt = prompt,
            temperature = self.temperature,
            top_p = 0.95,
            max_new_tokens = 2048,
            num_return_sequences = self.n_actions
        )
        
        if isinstance(outputs, str):
            outputs = [outputs]
        
        new_prompts = []
        
        # logging 
        for output in outputs:
            # parse the output
            output = parse_json_output(output)
            # logging
            logging.info(output_log.format(output=json.dumps(output, indent=4)))
            # append the new prompt
            new_prompts.append(output["new_system_prompt"].replace("\\n", "\n"))
        
        return False, new_prompts
    
    def fast_reward(self, state: PromptAlignState, action: PromptAlignAction, **kwargs) -> tuple[float, dict]:
        return 0, {}
    
    def reward(self, state: PromptAlignState, action: PromptAlignAction, **kwargs) -> float:
        # get the eval_dict directly from kwargs
        eval_dict = kwargs["eval_dict"]
        
        if len(eval_dict) == 0:
            return 0
        
        # calculate the reward by averaging the scores
        reward = sum([int(eval_dict[aspect]["score"]) for aspect in eval_dict]) / len(eval_dict)
        
        return reward
        
                                
                                     
            
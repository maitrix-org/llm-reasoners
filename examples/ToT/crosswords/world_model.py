import io
from typing import NamedTuple, List, Tuple
from reasoners import WorldModel, LanguageModel
from prompts.crosswords import * 
from utils import *
from reasoners.lm import OpenAIModel, Llama2Model, Llama3Model

CrosswordsState = Tuple[MiniCrosswordsEnv, List, dict]
CrosswordsAction = Tuple[str, float]


class CrosswordsWorldModel(WorldModel):
    """
    crosswords World Model
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    State: env, actions, info (of this state)
    """

    def __init__(self,
                 base_model: LanguageModel,
                 ) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt_status_cache = {}

    def init_state(self) -> list:
        ## input, output
        env = MiniCrosswordsEnv()
        env.reset(self.example)
        return (env, [], {})

    def is_terminal(self, state: CrosswordsState) -> bool:
        env, actions, info = state
        if len(info) == 0:
            return False
        return True
    
    def prompt_status(self, env):
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        for ans, data, status in zip(env.ans, env.data, env.status):
            # if status != 0: continue
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = value_prompt.format(input=line)
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                if isinstance(self.base_model, OpenAIModel):
                    eos_token_id = []
                elif isinstance(self.base_model, Llama2Model):
                    eos_token_id = ["\n"]
                elif isinstance(self.base_model, Llama3Model):
                    eos_token_id = ["\n\n", ".\n", ".\n\n","\n"]
                res = self.base_model.generate(prompt,
                                            num_return_sequences=1,
                                            stop=None,
                                            hide_input=True,
                                            do_sample=False,
                                            temperature=0,
                                            eos_token_id=eos_token_id).text[0].strip()

                #res = self.base_model.generate(prompt, num_return_sequences=1, stop=None).text[0]
                self.prompt_status_cache[prompt] = res
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1
        return count

    def step(self, state: CrosswordsState, action: CrosswordsAction) -> CrosswordsState:
        env, actions, info = state
        # back up current state
        board, status, steps, cur_ans = env.board.copy(), env.status.copy(), env.steps, env.ans.copy()
        new_state_actions = actions.copy()
        
        ## create a new state for step forward
        new_env = MiniCrosswordsEnv()
        new_env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)
        new_env.ans = cur_ans.copy()

        ## to next state
        obs, r, done, new_info = new_env.step(action[0])
        print('new action check', action, new_env.steps, new_env.status)
        count = self.prompt_status(env=new_env)
        new_state_actions.append(action)
        new_info = {'env_step': new_env.steps, 'actions': new_state_actions.copy(), 'info': new_info, 'count': count}
        new_state = (new_env, new_state_actions, new_info)

        return new_state
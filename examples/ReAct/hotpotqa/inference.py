import wikienv, wrappers
from reasoners import Reasoner, SearchConfig, WorldModel
from reasoners.algorithm import GreedySearch, GreedySearchResult
from reasoners.lm import Llama3Model
from reasoners.benchmark import Hotpotqaevaluator
import utils
import copy
import json
import fire
from typing import NamedTuple
from langchain.tools import StructuredTool

HotpotqaAction = str
HotpotqaExample = str

class HotpotqaState(NamedTuple):
    "The state of the Hotpotqa."
    step_idx: int
    last_state: str
    current_state: str
    buffered_action: HotpotqaAction

class ReActSearchConfig(SearchConfig):

    def __init__(self,
                 base_model,
                 temperature=0):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature

    def get_actions(self, state):
        inputs = state.current_state
        try:
            outputs = self.base_model.generate([inputs],
                                                hide_input=True,
                                                do_sample=True,
                                                max_new_tokens=512,
                                                temperature=self.temperature,
                                                eos_token_id=["Observ","Question"]).text[0]
        except:
            return ["exceed maxlength"]

        return [outputs]

    def reward(self, state, action, ans_finished):
        return 0
    
class ReActWorldModel(WorldModel[HotpotqaState, HotpotqaAction, HotpotqaExample]):
    def __init__(self,
                 base_model,
                 toolset,
                 max_steps: int = 8) -> None:
        self.base_model = base_model
        self.toolset = toolset
        self.max_steps = max_steps
        self.terminal = False

    def init_state(self):
        """Initialize the world model.
        :return: the initial state
        """
        current_state = self.prompt["ReAct"]+"Question: "+self.example+'\n'+"Thought 1: "
        return HotpotqaState(step_idx=1, last_state="", current_state=current_state, buffered_action="")

    def step(self, state, action):
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of BlocksWorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """

        state = copy.deepcopy(state)
        current_state = state.current_state
        step_idx = state.step_idx
        current_state = self.update_state(current_state, action, step_idx)
        new_buffered_action = action

        if action == "exceed maxlength":
            self.terminal = True

        if self.terminal == True:
            state = HotpotqaState(step_idx=self.max_steps, last_state=None,
                        current_state=None, buffered_action= "Finish[]")
            aux = {"ans_finished": self.terminal}
            return state, aux

        state = HotpotqaState(step_idx=step_idx+1, last_state=state.current_state,
                        current_state=current_state, buffered_action=new_buffered_action)
        aux = {"ans_finished": self.terminal}

        return state, aux

    def update_state(self, current_state: str, action: HotpotqaAction, step_idx) -> str:
        """Update the block states with the action.

        :param block_states: the current block states. Note that this argument is a string,
            and it's only a part of 'BWState'
        :param action: the action to take
        :return: the updated block states
        """
        try:
            thought, action = action.strip().split(f"\nAction {step_idx}: ")
        except:
            thought = action.strip().split('\n')[0]
            if step_idx != 1 and thought == "":
                self.terminal = True
                return None
            new_state = current_state + f" {thought}\nAction {step_idx}:"
            return new_state

        print(thought,action)
        # The case we need to use search tool
        if "Search" in action or "Lookup" in action or "Finish" in action:
            new_state = current_state + self.toolset[0].run({'env':self.base_model, 'step_idx': step_idx, 'action': action, 'thought': thought})
        
        return new_state

    def is_terminal(self, state: HotpotqaState) -> bool:
        if self.terminal == True:
            self.base_model.reset()
            self.terminal = False
            return True
        if utils.finished_check(state.buffered_action):
            self.base_model.reset()
            self.terminal = False
            return True
        if state.step_idx == self.max_steps:
            self.base_model.reset()
            self.terminal = False
            return True
        return False

def main(model_dir, llama_size="8B", prompt="examples/ReAct/hotpotqa/prompts/react.json", data_path="examples/ReAct/hotpotqa/data/hotpot_dev_v1_simplified.json"):
    
    base_model = Llama3Model(model_dir, llama_size, max_batch_size=1, max_seq_len=20000)

    search = StructuredTool.from_function(
        func=utils.webthink,
        name="Search",
        description="Action can be three types: \n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.\n(3) Finish[answer], which returns the answer and finishes the task.\nHere are some examples.\n",
        )    
    print(search.name)
    print(search.description)
    print(search.args)

    toolset = [search]
    
    with open(prompt) as f:
        prompt = json.load(f)
    
    evaluator = Hotpotqaevaluator(
        output_extractor=utils.retrieve_answer,
        answer_extractor=lambda x: x["answer"],
        data_path=data_path,
        toolset=toolset,
        init_prompt=prompt,
        disable_log=False,
        disable_tqdm=False)

    world_base_model = wikienv.WikiEnv()
    world_base_model = wrappers.HotPotQAWrapper(world_base_model, split="dev")
    world_base_model = wrappers.LoggingWrapper(world_base_model)

    reasoner = Reasoner(
        world_model=ReActWorldModel(world_base_model,toolset),
        search_config=ReActSearchConfig(base_model),
        search_algo=GreedySearch(7)
    )

    # run the reasoner
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4)

    print(accuracy)
    
if __name__ == "__main__":
    fire.Fire(main)

import wikienv, wrappers
from reasoners import Reasoner, SearchConfig, WorldModel, Tool
from reasoners.algorithm import GreedySearch, GreedySearchResult
from reasoners.lm import Llama3Model
from reasoners.benchmark import Hotpotqaevaluator
from reasoners.tools import wikisearch,wikilookup,wikifinish
import utils
import copy
import json
import fire
from typing import NamedTuple
from langchain.tools import StructuredTool

class HotpotQATools(NamedTuple):
   search: wikisearch
   lookup: wikilookup
   finish: wikifinish

HotpotqaAction = str
HotpotqaExample = str

class HotpotqaState(NamedTuple):
    "The state of the Hotpotqa."
    step_idx: int
    last_state: str
    current_state: str
    action: str

class HotpotqaSearchConfig(SearchConfig):

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
    
class HotpotqaWorldModel(WorldModel[HotpotqaState, HotpotqaAction, HotpotqaExample]):
    def __init__(self,
                 base_model,
                 max_steps: int = 8) -> None:
        self.base_model = base_model
        self.max_steps = max_steps
        self.terminal = False

    def init_state(self):
        """Initialize the world model.
        :return: the initial state
        """
        current_state = self.prompt["prefix"]+", and Action can be three types: \n"
        for idx,tool in enumerate(self.toolset):
            current_state += f"({str(idx+1)}) " + tool.description + "\n"
        current_state += "Here are some examples.\n" + "".join(self.prompt["examples"])
        current_state += "Question: "+self.example+'\n'+"Thought 1: "
        
        return HotpotqaState(step_idx=1, last_state="", current_state=current_state, action="")

    def step(self, state, action):
        "Take a step in the world model."

        state = copy.deepcopy(state)
        current_state = state.current_state
        step_idx = state.step_idx
        try:
            thought, action = action.strip().split(f"\nAction {step_idx}: ")
            if "Search" in action:
                new_state = current_state + HotpotQATools.wikisearch(env=self.base_model, step_idx=step_idx, action=action, thought=thought)
                print("Search tool is used")
            elif "Lookup" in action:
                new_state = current_state + HotpotQATools.wikilookup(env=self.base_model, step_idx=step_idx, action=action, thought=thought)
                print("Lookup tool is used")
            elif "Finish" in action:
                new_state = current_state + HotpotQATools.wikifinish(env=self.base_model, step_idx=step_idx, action=action, thought=thought)
                print("Finish tool is used")
            else:
                self.terminal = True
                aux = {"ans_finished": self.terminal}
                return None, aux
        except:
            thought = action.strip().split('\n')[0]
            if step_idx != 1 and thought == "":
                self.terminal = True
                aux = {"ans_finished": self.terminal}
                return None, aux
            new_state = current_state + f" {thought}\nAction {step_idx}:"

        current_state = new_state
        # print(thought,action)
        action = action

        if action == "exceed maxlength":
            self.terminal = True

        if self.terminal == True:
            state = HotpotqaState(step_idx=self.max_steps, last_state=None,
                        current_state=None, action= "Finish[]")
            aux = {"ans_finished": self.terminal}
            return state, aux

        state = HotpotqaState(step_idx=step_idx+1, last_state=state.current_state,
                        current_state=current_state, action=action)
        aux = {"ans_finished": self.terminal}

        return state, aux

    def is_terminal(self, state: HotpotqaState) -> bool:
        if self.terminal == True:
            self.base_model.reset()
            self.terminal = False
            return True
        if utils.finished_check(state.action):
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
    
    with open(prompt) as f:
        prompt = json.load(f)
    
    evaluator = Hotpotqaevaluator(
        output_extractor=utils.retrieve_answer,
        answer_extractor=lambda x: x["answer"],
        data_path=data_path,
        init_prompt=prompt,
        disable_log=False,
        disable_tqdm=False)

    world_base_model = wikienv.WikiEnv()
    world_base_model = wrappers.HotPotQAWrapper(world_base_model, split="dev")
    world_base_model = wrappers.LoggingWrapper(world_base_model)

    reasoner = Reasoner(
        world_model=HotpotqaWorldModel(world_base_model,toolset),
        search_config=HotpotqaSearchConfig(base_model),
        search_algo=GreedySearch(7)
    )

    # run the reasoner
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=False, num_shot=6)

    print(accuracy)
    
if __name__ == "__main__":
    fire.Fire(main)

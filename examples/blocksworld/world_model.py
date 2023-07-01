from typing import NamedTuple
from rap import WorldModel, LanguageModel
import utils

BWAction = str
class BWState(NamedTuple):
    step_idx: int
    blocks_state: str
    buffered_action: BWAction


class BlocksWorldModel(WorldModel[BWState, BWAction]):
    """
    Blocks World World Model
    State: (step_idx, block state, buffered action)
    Action: e.g. "put the red block on the green block"
    Special note about the state:
        the block state is updated every two actions. When there is a block in hand, the block state is not updated, but the action is buffered. With the next action, the block state is updated and the buffer is cleared.
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size=2) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> list:
        return []

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        state = state.copy()
        if state[1] == "":
            # if no action buffered, simply buffer the action
            state = state.copy()
            state[1] = action
            return state, {}
        
        buffered_action = state["buffered_action"]
        blocks_state = state["blocks_state"]
        step_idx = state["step_idx"]
        blocks_state = self.update_blocks(blocks_state, buffered_action)
        blocks_state = self.update_blocks(state, action)
        state = BWState(step_idx=step_idx+1, blocks_state=blocks_state, buffered_action="")
        return state, {}

    def update_blocks(self, block_states: str, action: BWAction) -> str:
        if "Pick" in action: 
            world_update_prompt = self.prompts["world_update_pickup"].format(block_states, action)
        elif "Unstack" in action:
            world_update_prompt = self.prompts["world_update_unstack"].format(block_states, action)
        elif "Put" in action:
            world_update_prompt = self.prompts["world_update_putdown"].format(block_states, action)
        elif "Stack" in action: 
            world_update_prompt = self.prompts["world_update_stack"].format(block_states, action)

        world_output = self.base_model.generate([world_update_prompt], num_return_sequences=1,
                                    eos_token="\n", hide_input=True).text[0]
        world_change = world_output.split("[CHANGE]")[-1]
        new_state = utils.apply_change(world_change, block_states)
        return new_state

    def is_terminal(self, state: GSM8kState) -> bool:
        pass

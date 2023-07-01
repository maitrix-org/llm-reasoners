"""The world model for the Blocksworld."""

from typing import NamedTuple
import utils
from rap import WorldModel, LanguageModel

BWAction = str
class BWState(NamedTuple):
    """The state of the Blocksworld.
    
    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    blocks_state: str
    buffered_action: BWAction


class BlocksWorldModel(WorldModel[BWState, BWAction]):
    """Blocks World World Model
    State: (step_idx, block state, buffered action)
    Action: e.g. "put the red block on the green block"
    Special note about the state:
        the block state is updated every two actions. When there is a block in hand, 
        the block state is not updated, but the action is buffered. With the next action, 
        the block state is updated and the buffer is cleared.
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
        """Initialize the world model.

        :return: the initial state
        """
        return BWState(step_idx=0, blocks_state=utils.
                       extract_init_state(self.example), buffered_action="")

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        """Take a step in the world model.
        
        :param state: the current state
        :param action: the action to take
        :return: the next state and an empty dict (placeholder)
        """
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
        """Update the block states with the action.

        :param block_states: the current block states
        :param action: the action to take
        :return: the updated block states
        """
        if "Pick" in action:
            world_update_prompt = self.prompt["world_update_pickup"].format(block_states, action)
        elif "Unstack" in action:
            world_update_prompt = self.prompt["world_update_unstack"].format(block_states, action)
        elif "Put" in action:
            world_update_prompt = self.prompt["world_update_putdown"].format(block_states, action)
        elif "Stack" in action:
            world_update_prompt = self.prompt["world_update_stack"].format(block_states, action)

        world_output = self.base_model.generate([world_update_prompt], num_return_sequences=1,
                                    eos_token="\n", hide_input=True).text[0]
        world_change = world_output.split("[CHANGE]")[-1]
        new_state = utils.apply_change(world_change, block_states)
        return new_state

    def is_terminal(self, state: BWState) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state["blocks_state"])[0]:
            return True
        return False

"""The world model for the Blocksworld."""

from typing import NamedTuple
import reasoners.benchmark.bw_utils as utils
from reasoners import WorldModel, LanguageModel
import copy

BWAction = str
class BWState(NamedTuple):
    """The state of the Blocksworld.
    
    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: BWAction


class BlocksWorldModel(WorldModel):
    """Blocks World World Model
    State: (step_idx, last_blocks_state, blocks_state, buffered_action)
    Action: e.g. "put the red block on the green block"
    Additional notes about the state:
        the block state is updated every two actions. When there is a block in hand, 
        the block state is not updated, but the action is buffered. With the next action, 
        the block state is updated and the buffer is cleared.
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 6,
                 batch_size=2) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> BWState:
        """Initialize the world model.

        :return: the initial state
        """
        return BWState(step_idx=0, last_blocks_state="", blocks_state=utils.
                       extract_init_state(self.example), buffered_action="")

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of BlocksWorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        buffered_action = state.buffered_action
        blocks_state = state.blocks_state
        step_idx = state.step_idx
        blocks_state = self.update_blocks(blocks_state, action)
        if state.buffered_action == "":
            # if no action buffered, buffer the action
            new_buffered_action = action
        else:
            # if action buffered, clear the buffer
            new_buffered_action = ""

        state = BWState(step_idx=step_idx+1, last_blocks_state=state.blocks_state,
                        blocks_state=blocks_state, buffered_action=new_buffered_action)
        return state, {"goal_reached": utils.goal_check(utils.extract_goals(self.example), blocks_state)}

    def update_blocks(self, block_states: str, action: BWAction) -> str:
        """Update the block states with the action.

        :param block_states: the current block states. Note that this argument is a string,
            and it's only a part of 'BWState'
        :param action: the action to take
        :return: the updated block states
        """
        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            raise ValueError("Invalid action")
        world_update_prompt = self.prompt[key].format(block_states, action.capitalize() + ".")
        world_output = self.base_model.generate([world_update_prompt],
                                    eos_token_id="\n", hide_input=True, temperature=0).text[0].strip()
        new_state = utils.apply_change(world_output, block_states)
        return new_state

    def is_terminal(self, state: BWState) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state.blocks_state)[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

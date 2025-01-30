"""The world model for the Math500."""

from typing import NamedTuple
from reasoners import WorldModel, LanguageModel
import copy
import re
from loguru import logger
import time

MathAction = str


class MathState(NamedTuple):
    """The state of the Math problem.
    """

    step_idx: int
    steps: list
    end: bool
    solution: str


class MathModel(WorldModel):
    """Math World Model
    State: (step_idx, last_problem_state, problem_state, buffered_action)
    Action: e.g. todo
    Additional notes about the state:
        todo
    """

    def __init__(
        self,
        base_model: LanguageModel,
        prm: LanguageModel,
        prompt: dict,
    ) -> None:
        super().__init__()

        self.base_model = base_model
        self.reward_model = prm
        self.prompt = prompt
        self.end = False
        self.question = None

    def init_state(self) -> MathState:
        """Initialize the world model.

        :return: the initial state
        """
        return MathState(step_idx=0, steps=[], end=self.end, solution="")

    @staticmethod
    def step_helper(state: MathState, action_list: MathAction) -> MathAction:

        step_num_to_take = state.step_idx + 1
        solution = state.solution

        if isinstance(action_list, list):
            action_list = action_list[0]
        if "## Step" in action_list:
            action = action_list.strip().split("## Step ")
        else:
            action = action_list.strip().split("Step ")

        num_actions = len(action)
        # Use regex to extract the number right after "Step" word in the last action.

        last_action_step_num = int(re.findall(r"\d+", action[-1])[0])
        try:
            first_action_step_num = int(re.findall(r"\d+", action[0])[0])
        except IndexError:
            first_action_step_num = int(re.findall(r"\d+", action[1])[0])

        if num_actions != last_action_step_num - first_action_step_num + 1:
            # There is a potential parsing error.
            # The last action should have the step number equal to the number of actions.

            all_actions = []
            for a in action:
                all_actions += a.split("Step ")

            action = all_actions

        is_terminal = False

        # The steps are done and the solution is available.
        # The solution is in $\\boxed{answer}$ format.
        # Extract the solution from the action.
        potential_actions = [
            a for a in action if a.strip().startswith(f"{step_num_to_take}:")
        ]
        future_actions_present = (
            len([a for a in action if a.strip().startswith(f"{step_num_to_take + 1}")])
            > 0
        )

        if ("$\\boxed{" in action[-1]) and not future_actions_present:
            solution = action[-1].split("$\\boxed{")
            solution = solution[-1].split("}$")
            solution = solution[0]

            is_terminal = True

        action_to_take = action[-1]

        if action_to_take.startswith("1:"):
            action_to_take = action_to_take.replace("1:", f"{step_num_to_take}:", 1)

        return action_to_take, is_terminal, solution

    def step(self, state: MathState, action_list: MathAction) -> tuple[MathState, dict]:
        """Take a step in the world model.

        state: the current state (see the docstring of MathModel)
        action_list: the action to take
        return: the next state and additional information cached for reward calculation
        """
        start = time.time()
        action_to_take, is_terminal, solution = MathModel.step_helper(
            state, action_list
        )

        logger.info(f"Action to take at step {state.step_idx+1}:\n{action_to_take}")

        state = copy.deepcopy(state)
        steps = state.steps + [
            action_to_take
        ]
        end = is_terminal

        state = MathState(
            step_idx=state.step_idx + 1,
            steps=steps,
            end=end,
            solution=solution,
        )

        logger.info(f"TIME: function step for Step {state.step_idx} took {time.time()-start} seconds")

        return state, {}

    def is_terminal(self, state: MathState) -> bool:
        return state.end

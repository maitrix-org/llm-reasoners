import gymnasium as gym
from typing import NamedTuple, Optional, Callable, Any
from reasoners import Environment
import time
from datetime import datetime

ActionGym = Any


class StateGym(NamedTuple):
    step_idx: int
    # action history used to reconstruct the env state for backtracking
    action_history: list[ActionGym]
    # gym observation objects
    last_obs: dict
    # outputs from env.step()
    current_obs: dict
    reward: float
    terminated: bool
    truncated: bool


class EnvironmentGym(Environment):
    """
    WorldModel, but for gym environments. Instead of being based off of a textual example, takes in a gym environment. An LLM will not be used for generating new states. The gym environment's step function takes care of that. 

    Attributes:
    - env (gym.Env): the gym environment
    - env_seed (int): the seed for the gym environment
    - max_steps (int): the maximum number of steps that can be taken until is_terminal cuts off the episode
    - obs_preprocessor (Optional[Callable[[dict], dict]]): optional function to process the observation returned from resetting/stepping the environment before it is stored into the state tuple
    - env_current_obs (dict): the current observation of the environment which is used to check if a passed in state is aligned with the environment's current state
    """

    def __init__(self, env: gym.Env, env_seed: int = 42, obs_preprocessor: Optional[Callable[[dict], dict]] = None, task_dir: str = None):
        self.env = env
        self.env_seed = env_seed
        self.obs_preprocessor = obs_preprocessor
        self.env_current_obs: dict = None
        self.task_dir = task_dir


    def log(self, text: str):
        current_time = datetime.now()
        formatted_time = current_time.strftime("[%Y%m%d] - %H:%M.%S")
        if text.startswith("\n"):
            text = f"\n{formatted_time}\n{text.lstrip()}"
        print(text)
        with open(f"{self.task_dir}/log.txt", "a+") as f:
            f.write(f"{text}\n")

    def init_state(self) -> StateGym:
        self.log("\ninit_state()")
        obs, env_info = self.env.reset(
            seed=self.env_seed)
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs

        return StateGym(step_idx=0, last_obs={}, current_obs=obs, action_history=[], reward=0, terminated=False, truncated=False)

    def align_env(self, state: StateGym):
        """
        Resets the environment and replays the action history to align an environment with the state.
        """
        start = time.time()
        self.env.reset(seed=self.env_seed)
        for action in state.action_history:
            self.env.step(action)
        end = time.time()
        self.log(f"align env time: {end - start}")


    def step(self, state: StateGym, action: ActionGym) -> tuple[StateGym, dict]:
        """
        Takes in a state and action and steps the environment. Should be noted that the environment may not be aligned with the state passed in. If the environment's current state (self.env_current_obs) is not the same as the state passed in, backtracking is needed. The basic implementation of this is rather naive, as it just resets the environment and replays the actions in the state's action_history list. Depending on the environment, there may be far more efficient ways to do so. 

        Args:
        - state (StateGym): the state to step from
        - action (ActionGym): the action to take from the state

        Returns:
        - next_state (StateGym): the next state after taking the action
        - aux (dict): used to pass the environment's reward to the search algorithm, which then passes it to the SearchConfig's reward function
        """

        self.log("\nstep()")

        if self.env_current_obs != state.current_obs:
            self.align_env(state)


        # TODO - add in code to handle time outs
        start = time.time()
        obs, reward, terminated, truncated, step_info = self.env.step(
            action)
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs
        end = time.time()
        self.log(f"env step time: {end - start}")

        next_state = StateGym(step_idx=state.step_idx + 1,
                              last_obs=state.current_obs, current_obs=obs,
                              action_history=state.action_history +
                              [action],
                              reward=reward, terminated=terminated, truncated=truncated)

        return next_state, {"env_reward": reward}

    def is_terminal(self, state: StateGym) -> bool:
        return state.terminated or state.truncated

import logging
import time
import pickle

from reasoners import SearchConfig, WorldModel, LanguageModel, Reasoner
from reasoners.algorithm import MCTS, BeamSearch
from reasoners.lm import OpenAIModel

from support import (
    get_env,
    reset_env,
    step_env,
    reset_and_replay_actions,
    create_logger,
    get_browser_action_set,
    get_clustered_action_proposals,
    get_parsed_evaluation_of_action_proposal,
)

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.env import BrowserEnv

from typing import NamedTuple


Action = str


class StateBrowsergym(NamedTuple):
    step_idx: int
    last_obs: dict  # instead of strings these will be obs objects
    current_obs: dict
    action_history: list[
        Action
    ]  # still need action history to be able to reconstruct the state for backtracking in mcts
    reward: float
    terminated: bool
    truncated: bool


class WorldModelBrowsergym(WorldModel):
    def __init__(
        self,
        env: BrowserEnv,
        env_seed: int = 16,
        logger: logging.Logger = None,
        max_steps: int = 20,
    ) -> None:
        super().__init__()
        self.env = env
        self.env_seed = env_seed
        self.env_current_obs = None
        self.logger = logger
        self.max_steps = max_steps

    def init_state(self) -> StateBrowsergym:
        obs, env_info = reset_env(self.env, self.env_seed, self.logger)
        self.env_current_obs = obs
        return StateBrowsergym(
            step_idx=0,
            last_obs={},
            current_obs=obs,
            action_history=[],
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def step(
        self, state: StateBrowsergym, action: Action
    ) -> tuple[StateBrowsergym, dict]:
        # if the current obs is not the same as the env_current_obs, then we need to set the env to the state for node expansion
        # only reset if we need to backtrack
        if state.current_obs != self.env_current_obs:
            self.env = reset_and_replay_actions(self.env, state.action_history)

        obs, reward, terminated, truncated, _ = step_env(self.env, action, self.logger)
        self.env_current_obs = obs

        next_state = StateBrowsergym(
            step_idx=state.step_idx + 1,
            last_obs=state.current_obs,
            current_obs=obs,
            action_history=state.action_history + [action],
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        print(
            f"NODE AT STEP {next_state.step_idx}\nTERMINATED: {terminated}\nTRUNCATED: {truncated}\nREWARD {reward}"
        )
        return next_state, {"goal_reached": bool(reward == 1.0)}

    def is_terminal(self, state: StateBrowsergym) -> bool:
        return state.terminated or state.truncated or state.step_idx >= self.max_steps


class SearchConfigBrowsergym(SearchConfig):
    def __init__(
        self,
        action_set: HighLevelActionSet,
        llm: LanguageModel,
        n_proposals: int = 10,
        proposal_temperature: float = 1.0,
        evaluation_temperature: float = 0.25,
        use_axtree: bool = True,
        use_html: bool = False,
        use_screenshot: bool = False,
        logger: logging.Logger = None,
    ) -> None:
        super().__init__()
        self.example = None  # technically doesn't need this. but i think it might have to be here due to the way it evaluates atm
        self.action_set = action_set
        self.llm = llm
        self.n_proposals = n_proposals
        self.proposal_temperature = proposal_temperature
        self.evlaution_temperature = evaluation_temperature
        self.use_axtree = use_axtree
        self.use_html = use_html
        self.use_screenshot = use_screenshot
        self.logger = logger

    def get_actions(self, state: StateBrowsergym) -> list[Action]:
        actions = get_clustered_action_proposals(
            state.current_obs,
            self.action_set,
            state.action_history,
            self.llm,
            n=self.n_proposals,
            temperature=self.proposal_temperature,
            logger=self.logger,
        )
        return actions

    def fast_reward(self, state: StateBrowsergym, action: Action) -> tuple[float, dict]:
        evaluation, info = get_parsed_evaluation_of_action_proposal(
            state.current_obs,
            action,
            self.action_set,
            state.action_history,
            self.llm,
            logger=self.logger,
        )
        return evaluation / 10, info

    def reward(
        self,
        state: StateBrowsergym,
        action: Action,
        intuition: float = None,
        self_eval: float = None,
        goal_reached: tuple[bool, float] = None,
    ) -> tuple[float, dict]:
        return self.fast_reward(state, action)


def run_task(task_name: str, seed: int = 16, llm: LanguageModel = None) -> bool:
    logger = create_logger(task_name)

    start_time = time.time()
    action_set = get_browser_action_set()
    action_history = []
    env = get_env(task_name, action_set, seed)

    # export OPENAI_API_KEY=[key]
    if llm is None:
        # llm = OpenAIModel(model="gpt-4o-mini")
        llm = OpenAIModel(model="gpt-4o", temperature=0.7)

    world_model = WorldModelBrowsergym(
        env=env, env_seed=seed, logger=logger, max_steps=20
    )
    search_config = SearchConfigBrowsergym(
        action_set=action_set,
        llm=llm,
        use_axtree=True,
        use_html=False,
        use_screenshot=False,
        logger=logger,
    )
    algorithm = MCTS(
        n_iters=10,
        depth_limit=4,  # depending on how long the task is, increase/decrease
        w_exp=2**0.5,
        uct_with_fast_reward=False,
        disable_tqdm=False,
        output_trace_in_each_iter=True,
    )
    # algorithm = BeamSearch(beam_size=3, max_depth=3) # beam is pretty nice to test on
    reasoner = Reasoner(world_model, search_config, algorithm)

    # print("sanity check")

    result = reasoner(
        ""
    )  # relies on a simulator for state - no explicit example needed

    end_time = time.time()

    task_completed = (
        result.terminal_state.reward == 1.0 if result.terminal_state else False
    )
    logger.info(f"<p><b>Task Completed: {task_completed}</b></p>")
    logger.info(f"<p><b>Time elapsed: {end_time - start_time}</b></p>")

    # pickle the result_rap object
    with open(f"./logs/{task_name}.pkl", "wb") as f:
        pickle.dump(result, f)

    env.close()

    return task_completed


# run_task("miniwob.login-user")
# run_task("miniwob.choose-date")
# run_task("miniwob.buy-ticket")

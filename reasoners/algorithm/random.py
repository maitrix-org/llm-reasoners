from typing import Generic, TypeVar, List, Tuple, NamedTuple
from .. import SearchAlgorithm, WorldModel, Reasoner, SearchConfig, State, Action
import random

class RandomShootingResult(NamedTuple):
    best_acc_reward: float
    best_trajectory: List[Tuple[Action, State, float]]
    trajectories: List[List[Tuple[Action, State, float]]]

class RandomShooting(SearchAlgorithm, Generic[State, Action]):
    """
    config.fast_reward is the prior to decide the order of exporation
    config.reward is the actual reward that decides the final result
    """

    def __init__(self, 
                 n_shoot: int = 10,  
                 max_depth: int = 10):
        self.n_shoot = n_shoot
        self.max_depth = max_depth
        self.trajactories = []

    def __call__(self, world, config):
        trajectories = []
        for _ in range(self.n_shoot):
            trajectory = []
            state = world.init_state()
            for _ in range(self.max_depth):
                actions = config.get_actions(state)
                # randomly sample an action
                action = random.choice(actions)
                reward, _ = config.reward(state, action)
                next_state, _ = world.step(state, action)
                state = next_state
                trajectory.append((action, state, reward))
                if world.is_terminal(state):
                    break
            trajectories.append(trajectory)
        
        best_acc_reward = -float('inf')
        best_trajectory = None

        for traj in trajectories:
            acc_reward = 0
            for _, _, reward in traj:
                acc_reward += reward
            if acc_reward > best_acc_reward:
                best_acc_reward = acc_reward
                best_trajectory = traj

        return RandomShootingResult(best_acc_reward, best_trajectory, trajectories)
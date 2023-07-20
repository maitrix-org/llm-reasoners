from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, RAPAgent, SearchConfig, State, Action
from typing import NamedTuple, List, Tuple, Callable, Any

def softmax(x: List[float], temperature: float) -> List[float]:
    # Implement softmax function
    e_x = np.exp(np.array(x) / temperature)
    return list(e_x / e_x.sum())

class BeamSearchResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: List[Tuple[Action, State]]


class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, 
                 beam_size: int, 
                 max_depth: int, 
                 sampling_strategy: str = 'argmax', # sampling strategy, argmax or softmax
                 temperature: float = 1.0, # temperature for softmax sampling
                 temperature_decay: float = 1.0, # temperature decay, default to no decay
                 reward_aggregator: Callable[[List[float]], float] = lambda x: sum(x)/len(x)
                ) -> None:
        # Initialize the BeamSearch class with a given beam size and maximum depth
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.sampling_method = sampling_method
        self.temperature = temperature
        self.reward_aggregator = reward_aggregator # aggregate the reward list into a single reward, default to be average
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x)) 
        return e_x / e_x.sum(axis=0)

    def _sample(self, beam):
        rewards = np.array([x[2] for x in beam])
        if self.sampling_strategy == 'argmax':
            indices = np.argmax(rewards)
        elif self.sampling_strategy == 'softmax':
            probs = self._softmax(rewards / self.temperature)
            indices = np.random.choice(len(probs), size=self.beam_size, p=probs)
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

        return [beam[i] for i in indices]
        

    def __call__(self, world: WorldModel[State, Action], config: SearchConfig[State, Action]):
        init_state = world.init_state()
        # Initialize current beam with initial state
        cur_beam = [([(None, init_state)], [], 0)]   # (trace, reward_list, reward)
        terminal_beam = []

        for _ in range(self.max_depth):
            new_beam = []
            for trace, reward_list, score in cur_beam:
                state = trace[-1][-1]
                if world.is_terminal(state) or len(trace) == self.max_depth:
                    terminal_beam.append((trace, reward_list, score))
                else:
                    for action in config.get_actions(state):
                        next_state = world.step(state, action)
                        reward = config.reward(state, action, next_state=next_state)
                        # Add new reward to list of rewards
                        new_reward_list = reward_list + [reward]
                        # Calculate the new reward
                        new_reward = self.reward_aggregator(new_reward_list)
                        new_beam.append((trace + [(action, next_state)], new_reward_list, new_reward))
            # Sort new beam by reward
            new_beam.sort(key=lambda x: x[2], reverse=True)
            # Sample from new beam
            cur_beam = self._sample(new_beam)

            # Decay the temperature
            self.temperature *= self.temperature_decay

        # Sort terminal beam by reward
        terminal_beam.sort(key=lambda x: x[2], reverse=True)
        best_result = terminal_beam[0]
        result = BeamSearchResult(
            terminal_state=best_result[0][-1][-1], 
            cum_reward=best_result[2],  # Use the precomputed cum_reward
            trace=best_result[0]
            )

        return result
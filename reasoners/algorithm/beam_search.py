from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, RAPAgent, SearchConfig, State, Action
from typing import NamedTuple, List, Tuple, Callable, Any
import numpy as np

class BeamSearchResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: List[Tuple[Action, State]]


class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, 
                 beam_size: int, 
                 max_depth: int, 
                 sampling_strategy: str = 'argmax', # sampling strategy, argmax or softmax
                 replace: bool = False, # whether to sample with replacement
                 temperature: float = 1.0, # temperature for softmax sampling
                 temperature_decay: float = 1.0, # temperature decay, default to no decay
                 reward_aggregator: Callable[[List[float]], float] = lambda x: sum(x)/len(x)
                ) -> None:
        # Initialize the BeamSearch class with a given beam size and maximum depth
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.sampling_strategy = sampling_strategy
        self.replace = replace
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.reward_aggregator = reward_aggregator # aggregate the reward list into a single reward, default to be average

        # if the temperature is set to 0, then we force the sampling strategy to be argmax
        if self.temperature < 1e-3:
            self.sampling_strategy = 'argmax'
        
        # if sampling strategy not in argmax or stochastic, just use argmax
        if self.sampling_strategy not in ['argmax', 'stochastic']:
            self.sampling_strategy = 'argmax'
    
    @staticmethod
    def softmax(x: List[float], temperature: float) -> List[float]:
        e_x = np.exp(np.array(x) / temperature)
        return list(e_x / e_x.sum())


    def _sample(self, beam):

        if self.sampling_strategy == 'argmax':
            # sort the beam by reward
            beam.sort(key=lambda x: x[2], reverse=True)
            # return the top k
            return beam[:self.beam_size]

        elif self.sampling_strategy == 'stochastic':
            rewards = np.array([x[2] for x in beam])

            if len(rewards) == 0:
                return []

            # sample size is the minimum of beam size and the length of the beam
            sample_size = min(self.beam_size, len(beam))
            # calculate the probability distribution
            probs = BeamSearch.softmax(rewards, self.temperature)
            # sample from the probability distribution without replacement
            indices = np.random.choice(len(probs), size=sample_size, p=probs, replace=self.replace)

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
                        next_state, aux = world.step(state, action)
                        reward = config.reward(state, action, **aux)
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
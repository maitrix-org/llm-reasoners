from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action
from typing import NamedTuple, List, Tuple, Callable, Any, Union, Optional
import numpy as np
import warnings
import random
from copy import deepcopy

class BeamSearchResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: List[Tuple[Action, State]]


class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, 
                 beam_size: int, 
                 max_depth: int, 
                 sampling_strategy: str = 'argmax', # sampling strategy, argmax or softmax
                 replace: Optional[bool] = None, # whether to sample with replacement
                 temperature: Optional[float] = None, # temperature for softmax sampling
                 temperature_decay: Optional[float] = None, # temperature decay, default to no decay
                 reject_sample: Optional[bool] = None, # whether to reject the samples with reward less than the reject_min_reward
                 reject_min_reward: Optional[float] = None, # the minimum reward to reject the sample
                 unbiased: Optional[bool] = None, # whether to use unbiased sampling
                 reward_aggregator: Union[Callable[[List[Any]], float], str] = 'cumulative', # how to aggregate the reward list
                 action_dedup: bool = False, # whether to deduplicate the actions
                 early_terminate: bool = True # whether to add to terminal beam if the action is terminal
                ) -> None:
        # Initialize the BeamSearch class
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.sampling_strategy = sampling_strategy
        self.replace = replace
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.reject_sample = reject_sample
        self.reject_min_reward = reject_min_reward
        self.unbiased = unbiased
        self.reward_aggregator = reward_aggregator
        self.action_dedup = action_dedup
        self.early_terminate = early_terminate

        # if the temperature is set to 0, then we force the sampling strategy to be argmax
        if self.temperature < 1e-3:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Temperature is set to 0, sampling strategy is forced to be argmax.")
        
        # if sampling strategy not in argmax or stochastic, just use argmax
        if self.sampling_strategy not in ['argmax', 'stochastic']:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Sampling strategy only supports argmax or stochastic, but got {self.sampling_strategy}. \
                            Sampling strategy is changed to argmax automatically.")
        
        # if early_terminate is set to False, we need to inform the user that we will return the beam instead of the best trace
        if not self.early_terminate:
            warnings.warn(f"early_terminate is set to True, BeamSearch will return the beam instead of the best trace.")

    
    @staticmethod
    def softmax(x: List[float], temperature: float, unbiased: bool = False, action_probs: Optional[List[float]] = None) -> List[float]:
        e_x = np.exp(np.array(x) / temperature)

        if unbiased and action_probs is not None:
            # adjust the values by the action_probs
            adjusted_values = [ n*p for n, p in zip(e_x, action_probs)]

            return [p / sum(adjusted_values) / max(1, len(adjusted_values)) for p in e_x]

        return list(e_x / e_x.sum())


    def _sample(self, beam):

        if self.sampling_strategy == 'argmax':
            # sort the beam by reward
            beam.sort(key=lambda x: x[2], reverse=True)
            if self.reject_sample:
                # reject the samples with reward less than the reject_min_reward
                beam = [x for x in beam if x[2] >= self.reject_min_reward]
            # return the top k
            return beam[:self.beam_size]

        elif self.sampling_strategy == 'stochastic':
            rewards = np.array([x[2] for x in beam])

            if len(rewards) == 0:
                return []

            # sample size is the minimum of beam size and the length of the beam
            sample_size = min(self.beam_size, len(beam))
            
            acc_action_probs = [x[3][0] for x in beam]
            cur_action_prob = [x[3][1] for x in beam]

            # calculate the probability distribution
            if self.unbiased:
                probs = BeamSearch.softmax(rewards, self.temperature, self.unbiased, action_probs=acc_action_probs)

            else:
                probs = BeamSearch.softmax(rewards, self.temperature, self.unbiased, action_probs=None)

            # reject the samples with reward less than the reject_min_reward
            if self.reject_sample:
                indexes, topk_beam_idx, iterate_cnt = list(range(len(probs))), [], 0
                cur_probs = deepcopy(probs)

                # get the upper bound of the reward
                reward_upper_bound = max(rewards)
                reward_upper_bound -= 1e-5 # to avoid the case where the reward is exactly the upper bound


                while len(topk_beam_idx) < sample_size and len(indexes) and iterate_cnt < 100:
                    iterate_cnt += 1
                    # sample an index
                    idx = random.choices(list(range(len(indexes))), weights=cur_probs)[0]
                    idx = indexes[idx]

                    if random.uniform(0,1) < cur_action_prob[idx] and \
                        rewards[idx] > min(self.reject_min_reward, reward_upper_bound):
                        
                        topk_beam_idx.append(idx)
                        indexes.remove(idx)
                    
                        if self.unbiased:
                            cur_probs = BeamSearch.softmax([rewards[i] for i in indexes], 
                                                            self.temperature,
                                                            self.unbiased,
                                                            action_probs=[acc_action_probs[i] for i in indexes])
                            
                        else:
                            cur_probs = BeamSearch.softmax([rewards[i] for i in indexes], self.temperature)

            else:
                topk_beam_idx = np.random.choice(len(probs), size=sample_size, p=probs, replace=self.replace)

            return [beam[i] for i in topk_beam_idx]
        

    def __call__(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], action_dedup: bool=False, return_beam: bool=False, early_terminate: bool=True, reward_strategy: str='last_iter'):
        init_state = world.init_state()
        # Initialize current beam with initial state
        cur_beam = [([(None, init_state)], [], 0)]   # (trace, reward_list, reward)
        terminal_beam = []

        for _ in range(self.max_depth):
            new_beam = []
            cache_for_dedup = set()

            for beam_item in cur_beam:
                trace, reward_list, _ = beam_item[:3]

                state = trace[-1][-1]
                if self.early_terminate and (world.is_terminal(state) or len(trace) == self.max_depth):
                    terminal_beam.append(beam_item)
                else:
                    actions = config.get_actions(state)

                    if self.action_dedup:
                        # deduplicate the actions
                        actions = [a for a in actions if a not in cache_for_dedup]
                        cache_for_dedup.update(actions)
                    
                    for action in actions:
                        next_state, aux = world.step(state, action)
                        
                        if self.unbiased and self.sampling_strategy == 'stochastic':
                            # the action should have action.action_prob
                            try:
                                reward, reward_aux = config.reward(state, action, **aux)
                                acc_action_prob = reward_aux['acc_action_prob']
                                cur_action_prob = reward_aux['cur_action_prob']
                            except:
                                raise ValueError(f"If unbiased stochastic sampling is used, \
                                                   please make sure the reward function returns \
                                                   a dictionary with keys 'acc_action_prob', which \
                                                   is the accumulated action probability, and \
                                                   'cur_action_prob', which is the current action probability.")
                        else:
                            reward = config.reward(state, action, **aux)

                        # Add new reward to list of rewards
                        new_reward_list = reward_list + [reward]
                        # Compute new reward

                        if self.reward_aggregator == 'cumulative' or self.reward_aggregator == 'accumulative':
                            self.reward_aggregator = lambda x: sum(x)
                        elif self.reward_aggregator == 'mean' or self.reward_aggregator == 'average':
                            self.reward_aggregator = lambda x: sum(x) / len(x)
                        elif isinstance(self.reward_aggregator, str) and self.reward_aggregator.startswith('last'):
                            self.reward_aggregator = lambda x: x[-1]
                        else:
                            # if the reward_aggregator is a string but not the above, raise error
                            if isinstance(self.reward_aggregator, str):
                                raise NotImplementedError(f"Reward aggregator {self.reward_aggregator} is not implemented.")

                        new_reward = self.reward_aggregator(new_reward_list)

                        if self.unbiased and self.sampling_strategy == 'stochastic':
                            new_beam.append((trace + [(action, next_state)], new_reward_list, new_reward, (acc_action_prob, cur_action_prob)))
                        else:
                            new_beam.append((trace + [(action, next_state)], new_reward_list, new_reward))


            # Sort new beam by reward
            new_beam.sort(key=lambda x: x[2], reverse=True)

            # Sample from new beam
            cur_beam = self._sample(new_beam)

            # Decay the temperature
            self.temperature *= self.temperature_decay
        
        if not self.early_terminate:
            # simply return the beam
            return cur_beam

        # Sort terminal beam by reward
        terminal_beam.sort(key=lambda x: x[2], reverse=True)
        best_result = terminal_beam[0]
        result = BeamSearchResult(
            terminal_state=best_result[0][-1][-1], 
            cum_reward=best_result[2],  # Use the precomputed cum_reward
            trace=best_result[0]
            )

        return result

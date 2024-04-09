from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example
from typing import NamedTuple, List, Tuple, Callable, Any, Union, Optional
import numpy as np
import warnings
import random
from copy import deepcopy
import itertools

class GreedySearchNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, 
                 state: State, 
                 action: Action, 
                 reward: float, 
                 parent: Optional['GreedySearchNode'] = None, 
                 children: Optional[List['GreedySearchNode']] = None
                ) -> None:
        
        self.id = next(GreedySearchNode.id_iter)  
        self.state = state
        self.action = action
        self.reward = reward
        self.parent = parent
        self.children = children if children is not None else []

    def add_child(self, child: 'GreedySearchNode'):
        self.children.append(child)
    
    def get_trace(self) -> List[Tuple[Action, State, float]]:
        """ Returns the sequence of actions and states from the root to the current node """
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.reward))
            node = node.parent
        # Reverse the path to get actions and states in order
        path = path[::-1]
        return path

class GreedySearchResult(NamedTuple):
    terminal_state: GreedySearchNode
    cum_reward: float
    tree: GreedySearchNode
    trace: List[Tuple[Action, State, float]]


class GreedySearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, 
                 max_depth: int, 
                 sampling_strategy: str = 'argmax', # sampling strategy, argmax or softmax
                 replace: Optional[bool] = None, # whether to sample with replacement
                 temperature: Optional[float] = None, # temperature for softmax sampling
                 temperature_decay: Optional[float] = None, # temperature decay, default to no decay
                 reject_sample: Optional[bool] = None, # whether to reject the samples with reward less than the reject_min_reward
                 reject_min_reward: Optional[float] = None, # the minimum reward to reject the sample
                 unbiased: Optional[bool] = None, # whether to use unbiased sampling
                 reward_aggregator: Union[Callable[[List[Any]], float], str] = 'last', # how to aggregate the reward list
                 action_dedup: bool = False, # whether to deduplicate the actions
                 early_terminate: bool = True, # whether to add to terminal beam if the action is terminal
                 return_beam: bool = False # whether to return the beam instead of the best trace
                ) -> None:
        # Initialize the GreedySearch class
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
        self.return_beam = return_beam

        # Initializing the reward_aggregator based on the provided argument
        self._initialize_reward_aggregator()

        # Post processing after initialization
        self._post_initialization()

    def _initialize_reward_aggregator(self):
        # how to aggregate the reward list
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
    
    def _post_initialization(self):
        # if the temperature is set to 0, then we force the sampling strategy to be argmax
        if self.temperature and self.temperature < 1e-4:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Temperature is set to 0, sampling strategy is forced to be argmax.")

        

        
        # if early_terminate is set to False, we need to inform the user that we will return the beam instead of the best trace
        if not self.early_terminate:
            self.return_beam = True
            warnings.warn(f"early_terminate is set to False, GreedySearch will return the beam instead of the best trace.")

    
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
            return beam[0]

        

    def __call__(self, world: WorldModel[State, Action, State], config: SearchConfig[State, Action, State]):
        # reset id
        GreedySearchNode.reset_id()
        
        init_state = world.init_state()
        # root node
        root_node = GreedySearchNode(state=init_state, action=None, reward=0.0)
        # Initialize current beam with initial state
        cur_beam = [(root_node, [], 0.0)] # (node, reward_list, cum_reward)
        terminal_beam = []

        for depth in range(self.max_depth + 1):
            # when depth == max_depth, we need to add the cur_beam to terminal_beam
            new_beam = []
            cache_for_dedup = set()

            print("--"*20)
            for beam_item in cur_beam:
                node, reward_list, _ = beam_item

                state = node.state
                if self.early_terminate and world.is_terminal(state):
                    terminal_beam.append(beam_item)
                else:
                    actions = config.get_actions(state)

                    if self.action_dedup:
                        # deduplicate the actions
                        actions = [a for a in actions if a not in cache_for_dedup]
                        cache_for_dedup.update(actions)
                        
                    elif depth == self.max_depth:
                        terminal_beam.append(beam_item)
                    
                    for action in actions:
                        next_state, aux = world.step(state, action)
                        
                        if self.unbiased and self.sampling_strategy == 'stochastic':
                            # the action should have action.action_prob
                            try:
                                fast_reward, fast_reward_aux = config.fast_reward(state, action)
                                reward, reward_aux = config.reward(state, action, **aux, **fast_reward_aux)
                                acc_action_prob = reward_aux['acc_action_prob']
                                cur_action_prob = reward_aux['cur_action_prob']
                            except:
                                raise ValueError(f"If unbiased stochastic sampling is used, \
                                                   please make sure the reward function returns \
                                                   a dictionary with keys 'acc_action_prob', which \
                                                   is the accumulated action probability, and \
                                                   'cur_action_prob', which is the current action probability.")
                        else:
                            fast_reward, fast_reward_aux = config.fast_reward(state, action)
                            reward = config.reward(state, action, **aux, **fast_reward_aux)

                            # if the reward is a tuple, then it is (reward, aux)
                            if isinstance(reward, tuple):
                                reward, reward_aux = reward

                        # Add new reward to list of rewards
                        new_reward_list = reward_list + [reward]

                        # Compute new reward
                        new_reward = self.reward_aggregator(new_reward_list)

                        # Create new node
                        new_node = GreedySearchNode(state=next_state, action=action, reward=reward, parent=node)

                        # Add new node to children of current node
                        node.add_child(new_node)

                        if self.unbiased and self.sampling_strategy == 'stochastic':
                            new_beam.append((new_node, new_reward_list, new_reward, (acc_action_prob, cur_action_prob)))
                        else:
                            new_beam.append((new_node, new_reward_list, new_reward))


            # Sort new beam by reward
            new_beam.sort(key=lambda x: x[2], reverse=True)

            # Sample from new beam
            if(len(new_beam)==0):
                break
            cur_beam = [self._sample(new_beam)]

            # Decay the temperature
            if self.temperature_decay:
                self.temperature *= self.temperature_decay
        
        if not self.early_terminate:
            # add the cur_beam to terminal_beam
            terminal_beam += cur_beam

        # Sort terminal beam by reward
        terminal_beam.sort(key=lambda x: x[2], reverse=True)

        if self.return_beam:
            # convert terminal_beam to a list of GreedySearchResult
            terminal_beam = [GreedySearchResult(
                                terminal_node=item[0],
                                cum_reward=item[2],  # Use the precomputed cum_reward
                                trace=item[0].get_trace(),
                                tree=root_node
                                ) for item in terminal_beam]
            
            return terminal_beam


        best_result = terminal_beam[0]
        result = GreedySearchResult(
            terminal_state=best_result[0],
            cum_reward=best_result[2],  # Use the precomputed cum_reward
            trace=best_result[0].get_trace(),
            tree=root_node
            )
        return result

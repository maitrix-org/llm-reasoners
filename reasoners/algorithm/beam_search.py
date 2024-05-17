from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example
from typing import NamedTuple, List, Tuple, Callable, Any, Union, Optional
import numpy as np
import warnings
import random
from copy import deepcopy
import itertools


class BeamSearchNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self,
                 state: State,
                 action: Action,
                 reward: float,
                 parent: Optional['BeamSearchNode'] = None,
                 children: Optional[List['BeamSearchNode']] = None
                 ) -> None:
        self.id = next(BeamSearchNode.id_iter)
        self.state = state
        self.action = action
        self.reward = reward
        self.parent = parent
        self.children = children if children is not None else []

    def add_child(self, child: 'BeamSearchNode'):
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


class BeamSearchResult(NamedTuple):
    terminal_node: BeamSearchNode
    terminal_state: State
    cum_reward: float
    tree: BeamSearchNode
    trace: List[Tuple[Action, State, float]]


class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, beam_size: int, max_depth: int, sampling_strategy: str = 'argmax',
                 replace: Optional[bool] = None, temperature: Optional[float] = None,
                 temperature_decay: Optional[float] = None, reject_sample: Optional[bool] = None,
                 reject_min_reward: Optional[float] = None, unbiased: Optional[bool] = None,
                 reward_aggregator: Union[Callable[[List[Any]], float], str] = 'last', action_dedup: bool = False,
                 early_terminate: bool = True, return_beam: bool = False, **kwargs) -> None:
        # Initialize the BeamSearch class
        super().__init__(**kwargs)
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
        self.return_beam = return_beam

        # Initializing the reward_aggregator based on the provided argument
        self._initialize_reward_aggregator()

        # Postprocessing after initialization
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

        # argmax = greedy = deterministic = topk
        if self.sampling_strategy in ['greedy', 'deterministic', 'topk']:
            self.sampling_strategy = 'argmax'

        # if sampling strategy not in argmax or stochastic, just use argmax
        if self.sampling_strategy not in ['argmax', 'stochastic']:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Sampling strategy only supports argmax or stochastic, but got {self.sampling_strategy}. \
                            Sampling strategy is changed to argmax automatically.")

        # if early_terminate is set to False, we need to inform the user that we will return the beam instead of the best trace
        if not self.early_terminate:
            self.return_beam = True
            warnings.warn(
                f"early_terminate is set to False, BeamSearch will return the beam instead of the best trace.")

    @staticmethod
    def softmax(x: List[float], temperature: float, unbiased: bool = False,
                action_probs: Optional[List[float]] = None) -> List[float]:
        e_x = np.exp(np.array(x) / temperature)

        if unbiased and action_probs is not None:
            # adjust the values by the action_probs
            adjusted_values = [n * p for n, p in zip(e_x, action_probs)]

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
                reward_upper_bound -= 1e-5  # to avoid the case where the reward is exactly the upper bound

                while len(topk_beam_idx) < sample_size and len(indexes) and iterate_cnt < 100:
                    iterate_cnt += 1
                    # sample an index
                    idx = random.choices(list(range(len(indexes))), weights=cur_probs)[0]
                    idx = indexes[idx]

                    if random.uniform(0, 1) < cur_action_prob[idx] and \
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

    def __call__(self, world: WorldModel[State, Action, State], config: SearchConfig[State, Action, State]):
        # reset id
        BeamSearchNode.reset_id()
        
        init_state = world.init_state()
        # root node
        root_node = BeamSearchNode(state=init_state, action=None, reward=0.0)
        # Initialize current beam with initial state
        cur_beam = [(root_node, [], 0.0)]  # (node, reward_list, cum_reward)
        terminal_beam = []

        for depth in range(self.max_depth + 1):
            # when depth == max_depth, we need to add the cur_beam to terminal_beam
            new_beam = []
            cache_for_dedup = set()

            for beam_item in cur_beam:
                node, reward_list, _ = beam_item[:3]

                state = node.state
                if self.early_terminate and world.is_terminal(state):
                    terminal_beam.append(beam_item)
                    
                else:
                    
                    if depth == self.max_depth:
                        terminal_beam.append(beam_item)
                        continue

                    if self.action_dedup:
                        # deduplicate the actions
                        actions = [a for a in actions if a not in cache_for_dedup]
                        cache_for_dedup.update(actions)

                    actions = config.get_actions(state)

                    for action in actions:
                        next_state, aux = world.step(state, action)

                        if self.unbiased and self.sampling_strategy == 'stochastic':
                            # the action should have action.action_prob
                            try:
                                fast_reward, fast_reward_aux = config.fast_reward(state, action)
                                reward, reward_aux = config.reward(state, action, **aux, **fast_reward_aux)
                                acc_action_prob = reward_aux['acc_action_prob']
                                cur_action_prob = reward_aux['cur_action_prob']
                            except KeyError:
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
                        # print(reward, new_reward_list, new_reward)

                        # Create new node
                        new_node = BeamSearchNode(state=next_state, action=action, reward=reward, parent=node)

                        # Add new node to children of current node
                        node.add_child(new_node)

                        if self.unbiased and self.sampling_strategy == 'stochastic':
                            new_beam.append((new_node, new_reward_list, new_reward, (acc_action_prob, cur_action_prob)))
                        else:
                            new_beam.append((new_node, new_reward_list, new_reward))

            # Sort new beam by reward
            # print(new_beam)
            new_beam.sort(key=lambda x: x[2], reverse=True)

            # Sample from new beam
            cur_beam = self._sample(new_beam)

            # Decay the temperature
            if self.temperature_decay:
                self.temperature *= self.temperature_decay

        if not self.early_terminate:
            # add the cur_beam to terminal_beam
            terminal_beam += cur_beam

        # Sort terminal beam by reward
        terminal_beam.sort(key=lambda x: x[2], reverse=True)

        if self.return_beam:
            # convert terminal_beam to a list of BeamSearchResult
            terminal_beam = [BeamSearchResult(
                                terminal_node=item[0],
                                terminal_state=item[0].state,
                                cum_reward=item[2],  # Use the precomputed cum_reward
                                trace=item[0].get_trace(),
                                tree=root_node
                                ) for item in terminal_beam]

            return terminal_beam
        
        if len(terminal_beam) == 0:
            return BeamSearchResult(
                terminal_state=None,
                terminal_node=None,
                cum_reward=0,
                trace=[],
                tree=root_node
            )
        
        best_result = terminal_beam[0]

        result = BeamSearchResult(
            terminal_state=best_result[0].state,
            terminal_node=best_result[0],
            cum_reward=best_result[2],  # Use the precomputed cum_reward
            trace=best_result[0].get_trace(),
            tree=root_node
        )

        return result
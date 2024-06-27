import torch
import heapq
import random
import pickle
import gzip
import numpy as np

import editdistance

class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.reset()

    def reset(self):
        self._buffer = {}

    def add(self, problem, plan, sample, log_reward):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if the plans have already existed in the problem
        if problem not in self._buffer:
            self._buffer[problem] = {
                "sentences": [],
                "exists": set(),
            }
        if plan in self._buffer[problem]["exists"]:
            return
        
        for buffer_item in self._buffer[problem]["sentences"]:
           
            if buffer_item[0] >= log_reward:
                return
            else:
                self._buffer[problem]["exists"].remove(buffer_item[1])
                self._buffer[problem]["sentences"].remove(buffer_item)
                heapq.heapify(self._buffer[problem]["sentences"])
                self._buffer[problem]["exists"].add(plan)
                heapq.heappush(
                    self._buffer[problem]["sentences"],
                    (
                        log_reward,
                        plan,
                        sample
                    ),
                )
                return
                
        self._buffer[problem]["exists"].add(plan)
        if len(self._buffer[problem]["sentences"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[problem]["sentences"],
                (
                           log_reward,
                            plan,
                            sample
                ),
            )
            self._buffer[problem]["exists"].remove(popped[1])
        else:
            heapq.heappush(
                self._buffer[problem]["sentences"],
                (
                    log_reward,
                    plan,
                    sample
                ),
            )

    

    def sample(self, batch_size, problem):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        if problem not in self._buffer:
            return None, None
        prompt_buffer = self._buffer[problem]["sentences"]
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        return [prompt_buffer[i][0] for i in idx], [prompt_buffer[i][1] for i in idx], [prompt_buffer[i][2] for i in idx],
        

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
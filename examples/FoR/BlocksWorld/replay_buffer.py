import torch
import heapq
import pickle
import gzip
import numpy as np

class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """
    def __init__(self, buffer_size, prb=True, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.sim_tolerance = sim_tolerance
        self.prb = prb
        self.reset()

    def reset(self):
        self._buffer = {}

    def add(self, problem, plan, sample, log_reward):
        if problem not in self._buffer:
            self._buffer[problem] = {
                "sentences": [],
                "exists": set(),
            }
        if plan in self._buffer[problem]["exists"]:
            return
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
            
        if len(self._buffer[problem]["sentences"]) > self.buffer_size:

            popped = heapq.heappop(self._buffer[problem]["sentences"])
            self._buffer[problem]["exists"].discard(popped[1])
  
    def sample(self, batch_size, problem):
        if problem not in self._buffer:
            return None, None, None
        prompt_buffer = self._buffer[problem]["sentences"]
        
        if self.prb:
            
            priorities  = [item[0] for item in prompt_buffer]
            priorities = torch.tensor(priorities, dtype=torch.float32) 

            priorities = priorities - torch.max(priorities)  

            probabilities = torch.exp(priorities) / torch.sum(torch.exp(priorities))

            idx = torch.multinomial(probabilities, batch_size, replacement=True)
        else:
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
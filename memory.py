
from collections import namedtuple
import random
from cv2 import error
import numpy as np
from experience import Experience
import time

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.latest_max_TD_error = 0
        self.push_during_optimize = False

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class HighestErrorMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.TD_errors = np.array([])
        self.latest_max_TD_error = 0
        self.push_during_optimize = False

    # Appends TD_error when adding elements
    def push(self, TD_error, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
            self.TD_errors = np.append(self.TD_errors, TD_error)
        else:       
            # Replaces lowest TD_error when memory is full
            lowest_TD_error_index = np.argmin(self.TD_errors)
            self.memory[lowest_TD_error_index] = Experience(*args)
            self.TD_errors[lowest_TD_error_index] = TD_error


    def sample(self, batch_size):
        positions = random.choices(range(len(self.memory)), k=batch_size)
        samples = np.array([self.memory[i] for i in positions])
        return samples, positions
    
    def __len__(self):
        return len(self.memory)

    def update_errors_in_memory(self, positions, errors):
        for i,e in zip(positions, errors):
            self.TD_errors[i] = abs(e)
    
class PrioritizedReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =  []
        self.priorities = []
        self.position = 0
        self.beta = 0
        self.memory_not_filled_before = True
        self.latest_max_TD_error = 0
        self.push_during_optimize = False

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = Experience(*args)

        # Gives the last element added priority 1 first time we reach full capacity. Due to the order we assign priorities
        # in the initial fill, we assign priorities to  
        # fills replay memory with memories that get max priority 1 
        if len(self.priorities) == self.capacity and self.memory_not_filled_before:
            self.priorities[-1] = 1
            self.memory_not_filled_before = False

        if len(self.memory) < self.capacity:
            self.priorities[self.position] = max(self.priorities[:-1], default=1)
        else:
            self.priorities[self.position] = max(self.priorities)

        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, alpha=0.6):
        #Get all probabilities
        sample_probs = self.get_probabilities(alpha)
        # Sample 32 Transitions from sample_prob weights
        sample_indices = random.choices(range(len(self.memory)), k=batch_size, weights=sample_probs)
        samples = np.array([self.memory[i] for i in sample_indices])
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices
    
    def __len__(self):
        return len(self.memory)

    # To sample a probability for each batch
    # The probability sample will have sum = 1
    def get_probabilities(self, alpha):
        scaled_priorities = np.array(self.priorities) ** alpha
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
    
    def get_importance(self, probabilities):
        importance = 1/len(self.memory) * 1/probabilities
        importance = importance ** self.beta
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, positions, errors, offset=0.1):
        for i,e in zip(positions, errors):
            self.priorities[i] = (abs(e) + offset)

class PrioritizedIRBMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =  []
        self.priorities = []
        self.position = 0
        self.beta = 0
        self.memory_not_filled_before = True
        self.latest_max_TD_error = 0
        self.push_during_optimize = False

    def push(self, prio_TD_error,  *args):
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
            self.priorities.append(prio_TD_error)
        else:       
            # Replaces lowest prio_TD_error when memory is full
            lowest_prio_TD_error_index = np.argmin(self.priorities)
            self.memory[lowest_prio_TD_error_index] = Experience(*args)
            self.priorities[lowest_prio_TD_error_index] = prio_TD_error
        
    def sample(self, batch_size, alpha=0.6):
        # Get all probabilities
        sample_probs = self.get_probabilities(alpha)
        # Sample 32 Transitions from sample_prob weights
        sample_indices = random.choices(range(len(self.memory)), k=batch_size, weights=sample_probs)
        samples = np.array([self.memory[i] for i in sample_indices])
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices
    
    def __len__(self):
        return len(self.memory)

    def get_probabilities(self, alpha):
        scaled_priorities = np.array(self.priorities) ** alpha
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
    
    def get_importance(self, probabilities):
        importance = 1/len(self.memory) * 1/probabilities
        importance = importance ** self.beta
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, positions, errors, offset=0.1):
        for i,e in zip(positions, errors):
            self.priorities[i] = (abs(e) + offset)
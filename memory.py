
from collections import namedtuple
import random
from cv2 import error
import numpy as np
from experience import Experience

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

# (st,at,rt,st+1)
# Slow-down is not due to it using objects, using Experience class instead of transition gives same speed
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        #self.memory[self.position] = Transition(*args)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

#(st,at,rt,st+1), |δt|
class PrioritizedReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory =  []
        self.priorities = []
        self.position = 0
        self.first = True
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = Experience(*args)
        # for testing add random.uniform(0.0, 1.0)
        if self.first == True:
            self.priorities[self.position] = 0
            self.first = False
        else:
            # At first this is as appended as none, but we want it to have the largest current memory priority
            # so we take the max of everything but last priority that is set to none (when initialized first time when replay memory is filling)
            self.priorities[self.position] = max(self.priorities[:-1])
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, priority_scale=1.0):
        #Get all probabilities
        sample_probs = self.get_probabilities(priority_scale)
        # Sample 32 Transitions from sample_prob weights
        sample_indices = random.choices(range(len(self.memory)), k=batch_size, weights=sample_probs)
        samples = np.array([self.memory[i] for i in sample_indices])
        #samples = np.array(self.memory)[sample_indices] # very big sinner, never make copies and then takes indices
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices
    
    def __len__(self):
        return len(self.memory)

    # To sample a probability for each batch
    # The probability sample will have sum = 1
    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
    
    def get_importance(self, probabilities):
        importance = 1/len(self.memory) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    # Not this
    def set_priorities(self, positions, errors, offset=0.1):
        for i,e in zip(positions, errors):
            self.priorities[i] = abs(e) + offset

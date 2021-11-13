from collections import namedtuple

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

class Experience(object):
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def convert_to_named_tuple(self):
        return Transition(self.state, self.action, self.next_state, self.reward)
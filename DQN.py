import copy
from collections import namedtuple
from itertools import count
import math
import random
from cv2 import error
import numpy as np 
import time

import gym
import logger

from wrappers import *
from memory import ReplayMemory, PrioritizedReplay
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from experience import Experience


Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def optimize_model_prio(incentive=False):
    if len(memory) < BATCH_SIZE:
        return
    transitions, importance, positions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    tmp = []
    for e in transitions:
        tmp.append(e.convert_to_named_tuple())
    # Unzip the Tuple into a list
    #batch = Transition_p(*zip(*transitions))

    #(Get it to be [('state', 'reward'), (1, 2), (1,2 )] etc
    batch = Transition(*zip(*tmp))
    # Unpack all the actions and rewards
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')
    

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
     #state_action_values - expected_state_action_values.unsqueeze(1)
    importance = torch.from_numpy(importance).to('cuda')
    importance = torch.unsqueeze(importance, dim=1)
    #errors = (state_action_values - expected_state_action_values.unsqueeze(1).detach()).pow(2) * importance

    errors = (state_action_values - expected_state_action_values.unsqueeze(1).detach())
    weighted_loss = importance * errors
    loss = weighted_loss.mean()
    updated_weights = (weighted_loss + 1e-6).data.cpu().numpy()
    memory.set_priorities(positions, updated_weights.flatten().tolist())
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def train(env, n_episodes, render=False):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            # Push the memory to the list
            memory.push(state, action.to('cuda'), next_state, reward.to('cuda'))

            if steps_done % INCENTIVE_UPDATE == 0:
                incentive_memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))

            state = next_state

            # Uptimize the model after X timesteps
            if steps_done > INITIAL_MEMORY:
                #optimize_model()
                optimize_model_prio()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            # Calculate Running Reward to check if solved
            episode_reward_history.append(total_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            if done:
                break

        if episode % 20 == 0:
            logger.logkv("episode_reward", total_reward)
            logger.logkv("running_reward", running_reward)
            logger.logkv("episode", episode)
            logger.logkv("steps_done", steps_done)
            logger.dumpkvs()

        if episode % 100 == 0:
            model_name = "dqn_pong_test_per_model"
            print("Saved Model as : {}".format(model_name))
            torch.save(policy_net, model_name)
    env.close()
    return

def visualize(env, n_episodes, policy, render=True):
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_breakout_video', force=True)
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    INCENTIVE_SIZE = 100
    INCENTIVE_UPDATE = 1000
    DEBUG = 10
    episode_reward_history = []

    # Setup logging for the model
    logger.set_level(DEBUG)
    dir = "pong-test"
    logger.configure(dir=dir)
    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    # create networks
    policy_net = DQNbn(n_actions=env.action_space.n).to(device)
    target_net = DQNbn(n_actions=env.action_space.n).to(device)
    #policy_net = torch.load("dqn_pong_model")
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # initialize replay memory
    #memory = ReplayMemory(MEMORY_SIZE)
    memory = PrioritizedReplay(MEMORY_SIZE)
    incentive_memory = IncentiveReplay(INCENTIVE_SIZE)
    # train model
    train(env, 4000000)

    # Load and test model
    #
    #visualize(env, 1, policy_net, render=True)


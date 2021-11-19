import copy
from collections import namedtuple
from itertools import count
import math
import random
from cv2 import error
import numpy as np 
import time

import gym
from logger import Logger

from wrappers import *
from memory import ReplayMemory, PrioritizedReplay
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from experience import Experience
import time
from logger import configure

Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global steps_done
    sample = random.random()
    # eps_end is the lowest eps can get. 0.02 + 0.0000001 = 0.02 (lowerbound)
    # math.exp(-1. * steps_done / EPS_DECAY) = number between 0.999 (when value in math.exp close to 0) or 0.000001
    # when values are higher
    # multiplied by eps_start-eps_end (1 - 0.02) = 0.998
    # so 0.998 * decreasing value torwards 0 (close to starting at 1, moving torwards 0)
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    #print(eps_threshold)
    steps_done += 1

    # Increase the beta variable torwards 1 during training. The beta variable determines priority importance. 
    if ORM_PER:
        ORM.beta = min(INITIAL_BETA + (1 - 1 * math.exp(-1. * steps_done / EPS_DECAY)), 1)
    if IRM_PER:
        IRM.beta = min(INITIAL_BETA + (1 - 1 * math.exp(-1. * steps_done / EPS_DECAY)), 1)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def optimize_model(memory):
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
    tmp = []
    for e in transitions:
        tmp.append(e.convert_to_named_tuple())
    
    batch = Transition(*zip(*tmp))

    
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
    
    beta = 1
    TD_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1).detach())
    
    # smooth_l1_loss from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/smooth_l1_loss.py
    cond = TD_errors < beta
    TD_errors = torch.where(cond, 0.5 * TD_errors ** 2 / beta, TD_errors - 0.5 * beta)
    
    loss = TD_errors.mean()

    # Push sample with higest TD_error to IRM
    if(OPTIMIZE_MODEL_PUSH):
        highest_TD_error_index = torch.argmax(TD_errors)
        highest_TD_error = torch.max(TD_errors)
        TD_sample = tmp[highest_TD_error_index]
        # Garanteed push every IRM_PUSH_FREQ
        if(steps_done % IRM_PUSH_FREQ):
            IRM.push(TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        elif(highest_TD_error > memory.latest_max_TD_error):
            IRM.push(TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        
        memory.latest_max_TD_error = highest_TD_error
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def optimize_model_prio(memory):
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


    logger.logkv("positions", positions)
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
    
    importance = torch.from_numpy(importance).to('cuda')
    importance = torch.unsqueeze(importance, dim=1)
    beta = 1
    TD_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1).detach())
    
    # smooth_l1_loss from https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/smooth_l1_loss.py
    cond = TD_errors < beta
    TD_errors = torch.where(cond, 0.5 * TD_errors ** 2 / beta, TD_errors - 0.5 * beta)


    # Push sample with higest TD_error to IRM
    if(PRIO_OPTIMIZE_MODEL_PUSH):
        highest_TD_error_index = torch.argmax(TD_errors)
        highest_TD_error = torch.max(TD_errors)
        TD_sample = tmp[highest_TD_error_index]
        # Garanteed push every IRM_PUSH_FREQ
        if(steps_done % IRM_PUSH_FREQ):
            IRM.push(TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        elif(highest_TD_error > memory.latest_max_TD_error):
            IRM.push(TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        
        memory.latest_max_TD_error = highest_TD_error

    weighted_loss = importance * TD_errors
    logger.logkv("weighted_loss", weighted_loss.flatten().tolist())

    loss = weighted_loss.mean()
    logger.logkv("loss", loss.flatten().tolist())
    
    # detaches updated weights (dosen't detach loss), updated weights will be converted to list to prioritize,
    # so dosen't make sense to backpropagate
    updated_weights = (weighted_loss + 1e-6).data.cpu().detach().numpy()
    memory.set_priorities(positions, updated_weights.flatten().tolist())
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
    start_time = time.perf_counter()
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
                break

            reward = torch.tensor([reward], device=device)
            # Push the memory to the list
            #memory.push(state, action.to('cuda'), next_state, reward.to('cuda'))

            # Push memory to ORM every step
            # Memories are pushed to IRM every nth step in RANDOM IRM model
            # In other models, memories are pushed to IRM after TD_error calculations in prio_optimize_model and optimize_model
            ORM.push(state, action.to('cuda'), next_state, reward.to('cuda'))
            if(RANDOM_IRM):
                if steps_done % IRM_PUSH_FREQ == 0:
                    IRM.push(state, action.to('cuda'), next_state, reward.to('cuda'))
            
            state = next_state

            # Optimize the model after replay memory have been filled to INITIAL_MEMORY
            # Implement INITIAL_MEMORY for IRM? atm we implicitly it have reached a batch size worth of memories
            if steps_done > INITIAL_MEMORY:
                # ORM will be used for an gradient update every step, IRM
                # will be used every IRM_UPDATES_FREQ
                if(ORM_PER):
                    optimize_model_prio(ORM)
                else:
                    optimize_model(ORM)
                
                if steps_done % IRM_UPDATES_FREQ == 0:
                    if(IRM_PER):
                        optimize_model_prio(IRM)
                    else:
                        optimize_model(IRM)
                
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            # Calculate Running Reward to check if solved
            episode_reward_history.append(total_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            if done:
                break

        if episode % 1 == 0:
            logger.logkv("episode_reward", total_reward)
            logger.logkv("running_reward", running_reward)
            logger.logkv("episode", episode)
            logger.logkv("steps_done", steps_done)
            average_episode_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            logger.logkv("episode time", average_episode_time)
            logger.dumpkvs()

        if episode % 100 == 0:
            model_name = "dqn_pong_final_test_per_model"
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
    INITIAL_BETA = 0.4
    TARGET_UPDATE = 1000
    RENDER = True
    lr = 1e-4
    #INITIAL_MEMORY = 32
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    DEBUG = 10
    IRM_UPDATES_FREQ = 1000
    IRM_PUSH_FREQ = 100
    
    # Flag for HIGEST_ERROR_IRM implementation
    # Makes either optimize_model or prio_optimize_model push the highest TD_error memory to IRM
    # every IRM_PUSH_FREQ
    OPTIMIZE_MODEL_PUSH = False
    PRIO_OPTIMIZE_MODEL_PUSH = False

    # Model Flags 
    ORM_PER = False
    IRM_PER = False

    # Model Configurations
    RANDOM_IRM = True
    HIGHEST_ERROR = False
    HIGHEST_ERROR_PER = False
    DEBUG_MODEL = False

    if RANDOM_IRM:
        print("Model Configuration: RANDOM_IRM")
    
    if HIGHEST_ERROR:
        print("Model Configuration: HIGHEST_ERROR")
        OPTIMIZE_MODEL_PUSH = True

    if HIGHEST_ERROR_PER:
        print("Model Configuration: HIGHEST_ERROR with PER ORM")
        ORM_PER = True
        PRIO_OPTIMIZE_MODEL_PUSH = True

    if DEBUG_MODEL:
        print("Model Configuration: DEBUG_MODEL")
        ORM_PER = True
        IRM_PER = True
    
    episode_reward_history = []

    # Setup logging for the model
    dir = "debug"

    logger = configure(dir)
    logger.set_level(DEBUG)

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
    ORM = ReplayMemory(MEMORY_SIZE)
    IRM = ReplayMemory(MEMORY_SIZE)
    
    if(ORM_PER):
        print("Initialized ORM with Prioritized Experience Replay")
        ORM = PrioritizedReplay(MEMORY_SIZE)
    if(IRM_PER):
        print("Initialized IRM with Prioritized Experience Replay")
        IRM = PrioritizedReplay(MEMORY_SIZE)


    # train model
    train(env, 4000000, RENDER)

    # Load and test model
    #
    #visualize(env, 1, policy_net, render=True)


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
from memory import ReplayMemory, PrioritizedReplay, HighestErrorMemory, PrioritizedIRBMemory
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

def save_params():
    import json

    data = {}
    data['params'] = []
    data['params'].append({
    'BATCH_SIZE': BATCH_SIZE,
    'GAMMA': GAMMA,
    'EPS_START': EPS_START,
    'EPS_END': EPS_END,
    'EPS_DECAY': EPS_DECAY,
    'INITIAL_BETA': INITIAL_BETA,
    'TARGET_UPDATE': TARGET_UPDATE,
    'lr': lr,
    'INITIAL_MEMORY': INITIAL_MEMORY,
    'IRB_MEM_SIZE': IRB_MEMORY_SIZE,
    'ORB_MEM_SIZE': ORB_MEMORY_SIZE,
    'IRB_UPDATES_FREQ': IRB_UPDATES_FREQ,
    'IRB_PUSH_FREQ': IRB_PUSH_FREQ
    })

    with open(RESULTS_DIR + '.json', 'w+') as json_out:
        json.dump(data, json_out)

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
    if ORB_PER:
        ORB.beta = min(INITIAL_BETA + (1 - 1 * math.exp(-1. * steps_done / EPS_DECAY)), 1)
    if IRB_PER:
        IRB.beta = min(INITIAL_BETA + (1 - 1 * math.exp(-1. * steps_done / EPS_DECAY)), 1)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return
    
    # If we are running a highest error model and we are not the ORB (who have the push_during_optimize flag set to true), i.e. we are running HighestErrorMemory
    if (HIGHEST_ERROR or HIGHEST_ERROR_PER) and not memory.push_during_optimize:
        transitions, positions = memory.sample(BATCH_SIZE)
    else:
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
    
    # update errors in our td_error IRB for both HIGHEST_ERROR or HIGHEST_ERROR_PER 
    # since they both use optimize_error to update their IRB
    if((HIGHEST_ERROR or HIGHEST_ERROR_PER) and not memory.push_during_optimize):
        memory.update_errors_in_memory(positions, TD_errors)

    loss = TD_errors.mean()

    # Push sample with higest TD_error to IRB
    # only ORB should do this
    if memory.push_during_optimize and (HIGHEST_ERROR or HIGHEST_ERROR_PER):
        abs_TD_errors = torch.abs(TD_errors)
        highest_TD_error_index = torch.argmax(abs_TD_errors)
        highest_TD_error = torch.max(abs_TD_errors).item()

        TD_sample = tmp[highest_TD_error_index]

        # guaranteed push every IRB_PUSH_FREQ
        if steps_done % IRB_PUSH_FREQ:
            IRB.push(highest_TD_error, TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        elif highest_TD_error > memory.latest_max_TD_error:
            IRB.push(highest_TD_error, TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        
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


    # Push sample with higest TD_error to IRB (Highest Error Model)
    if memory.push_during_optimize and (HIGHEST_ERROR or HIGHEST_ERROR_PER):
        abs_TD_errors = torch.abs(TD_errors)
        highest_TD_error_index = torch.argmax(abs_TD_errors)
        highest_TD_error = torch.max(abs_TD_errors).item()
        
        TD_sample = tmp[highest_TD_error_index]
        # Garanteed push every IRB_PUSH_FREQ
        if steps_done % IRB_PUSH_FREQ:
            IRB.push(highest_TD_error, TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        elif highest_TD_error > memory.latest_max_TD_error:
            IRB.push(highest_TD_error, TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        
        memory.latest_max_TD_error = highest_TD_error

        
        memory.latest_max_TD_error = highest_TD_error
    
    weighted_loss = importance * TD_errors
    # detaches updated weights (dosen't detach loss), updated weights will be converted to list to prioritize,
    # so dosen't make sense to backpropagate
    updated_weights = (weighted_loss + 1e-6).data.cpu().detach().numpy()
    memory.set_priorities(positions, updated_weights.flatten().tolist())

    # Push sample with higest TD_error * importance to IRB (PRIORITIZED_IRB)
    if memory.push_during_optimize and PRIORITIZED_IRB:
        abs_prio_TD_errors = torch.abs(weighted_loss + 1e-6)
        highest_prio_TD_error_index = torch.argmax(abs_prio_TD_errors)
        highest_prio_TD_error = torch.max(abs_prio_TD_errors).item()
        
        TD_sample = tmp[highest_prio_TD_error_index]
        # Garanteed push every IRB_PUSH_FREQ
        if steps_done % IRB_PUSH_FREQ:
            IRB.push(highest_prio_TD_error, TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)
        elif highest_prio_TD_error > memory.latest_max_TD_error:
            IRB.push(highest_prio_TD_error, TD_sample.state, TD_sample.action, TD_sample.next_state, TD_sample.reward)

    logger.logkv("weighted_loss", weighted_loss.flatten().tolist())

    loss = weighted_loss.mean()
    logger.logkv("loss", loss.flatten().tolist())
    
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

            if done and info["ale.lives"] == 0:
                break
            else:
                next_state = get_state(obs)

                

            reward = torch.tensor([reward], device=device)

            # Push memory to ORB every step
            # Memories are pushed to IRB every nth step in RANDOM IRB model
            # In other models, memories are pushed to IRB after TD_error calculations in optimize_model_prio and optimize_model
            ORB.push(state, action.to('cuda'), next_state, reward.to('cuda'))
            if(RANDOM_IRB):
                if steps_done % IRB_PUSH_FREQ == 0:
                    IRB.push(state, action.to('cuda'), next_state, reward.to('cuda'))
            
            state = next_state

            # Optimize the model after replay memory have been filled to INITIAL_MEMORY
            if steps_done > INITIAL_MEMORY:
                # ORB will be used for an gradient update every step, IRB
                # will be used every IRB_UPDATES_FREQ
                if ORB_PER:
                    optimize_model_prio(ORB)
                else:
                    optimize_model(ORB)
                
                if steps_done % IRB_UPDATES_FREQ == 0 and not (IRB_PER or HIGHEST_ERROR or PRIORITIZED_IRB):
                    if IRB_PER:
                        optimize_model_prio(IRB)
                    else:
                        optimize_model(IRB)
                
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done and info["ale.lives"] == 0:
                break
        
        
        # Calculate Running Reward to check if solved
        episode_reward_history.append(total_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        if episode % 1 == 0:
            logger.logkv("episode_reward", total_reward)
            logger.logkv("running_reward", running_reward)
            logger.logkv("episode", episode)
            logger.logkv("steps_done", steps_done)
            average_episode_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            logger.logkv("episode time", average_episode_time)
            logger.dumpkvs()

        if episode % 1000 == 0:
            print("Saved Model as : {}".format(MODEL_NAME))
            torch.save(policy_net, MODEL_NAME)
    env.close()
    return

def visualize(env, n_episodes, policy, render=True):
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_RoadRunner_video', force=True)
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
    TARGET_UPDATE = 10000
    RENDER = False
    lr = 1e-4
    #INITIAL_MEMORY = 32
    INITIAL_MEMORY = 50000
    ORB_MEMORY_SIZE = 2 * INITIAL_MEMORY
    IRB_MEMORY_SIZE = 1000
    DEBUG = 10
    IRB_UPDATES_FREQ = 500
    IRB_PUSH_FREQ = 50

    # Save Configurations
    RESULTS_DIR = "Breakout-duel-prio-irm-results"
    MODEL_NAME = "Breakout_duel_prio_irm_model"

    # Initialize Model Flags     
    # Has two effects. 1. Sets the beta variable for PER. 2. Does so prio_optimize_model is used rather than optimize_model.
    # Will be changed automatically depending on the model configuration flag
    ORB_PER = False
    IRB_PER = True

    # Model Configurations
    # No IRB if you don't want to test with any incentive replay buffer i.e. standard DQN or Dueling DQN
    NO_IRB = False
    RANDOM_IRB = False
    HIGHEST_ERROR = False
    PRIORITIZED_IRB = True
    HIGHEST_ERROR_PER = False
    

    # DUELING DQN
    DUELING_DQN = True

    if RANDOM_IRB:
        print("Model Configuration: RANDOM_IRB")
    
    if HIGHEST_ERROR:
        print("Model Configuration: HIGHEST_ERROR")
    
    if PRIORITIZED_IRB:
        print("Model Configuration: PRIORITIZED_IRB")
    
    episode_reward_history = []


    # Setup logging for the model
    logger = configure(RESULTS_DIR)
    logger.set_level(DEBUG)

    # create environment
    env = gym.make("BreakoutNoFrameskip-v4")
    env = make_env(env)
    # create networks
    policy_net = DQNbn(n_actions=env.action_space.n).to(device)
    target_net = DQNbn(n_actions=env.action_space.n).to(device)

    if(DUELING_DQN):
        policy_net = DuelingDQN(env.action_space.n, "cuda").to(device)
        target_net = DuelingDQN(env.action_space.n, "cuda").to(device)

    #policy_net = torch.load("dqn_RoadRunner_model")
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # Default ORB and IRB is standard experience replay. 
    # Changed during configurations to fit each incentive replay model variant. 
    ORB = ReplayMemory(ORB_MEMORY_SIZE)
    IRB = ReplayMemory(IRB_MEMORY_SIZE)
    
    if NO_IRB:
        print("Initialized with no IRB, ie. standard experience replay.")
        IRB = None
    
    if ORB_PER:
        print("Initialized ORB with Prioritized Experience Replay")
        ORB = PrioritizedReplay(ORB_MEMORY_SIZE)
        
    if IRB_PER:
        print("Initialized IRB with Prioritized Experience Replay")
        IRB = PrioritizedReplay(IRB_MEMORY_SIZE)

    if HIGHEST_ERROR:
        ORB.push_during_optimize = True
        IRB = HighestErrorMemory(IRB_MEMORY_SIZE)

    if PRIORITIZED_IRB:
        ORB.push_during_optimize = True
        IRB = PrioritizedIRBMemory(IRB_MEMORY_SIZE)

    # train model
    save_params()
    train(env, 4000000, RENDER)

    # Load and test model
    #visualize(env, 1, policy_net, render=True)



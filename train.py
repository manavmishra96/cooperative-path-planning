#train.py

import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import cv2
from collections import defaultdict
import matplotlib.animation as animation
from time import time as t
from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter

from env import PathPlan, agent
from actorcritic import Memory
from constants import CONSTANTS
import torch
from PPO import PPO
import pickle
import warnings

np.set_printoptions(threshold = np.inf, linewidth = 1000 ,precision=3, suppress=True)
warnings.filterwarnings('ignore')

const = CONSTANTS()

directory = const.directory
print (directory)
try:
    os.makedirs(directory, exist_ok = True)
    os.makedirs(os.path.join(directory, "images"), exist_ok = True)
    os.makedirs(os.path.join(directory, "train_graphs_smooth"), exist_ok = True)
    os.makedirs(os.path.join(directory, "checkpoints"), exist_ok = True)
    os.makedirs(os.path.join(directory, "renders"), exist_ok = True)
except OSError as error:
    print("Directory '%s' can not be created")
        
NUM_EPISODES = const.num_episode
LEN_EPISODES = const.len_episode
UPDATE_TIMESTEP = 1000
NUM_AGENTS = const.NUM_AGENTS
NUM_IMU = const.NUM_IMU

curState = []
newState= []
reward_history = []
agent_history_dict = defaultdict(list)
totalViewed = []
dispFlag = False
keyPress = 0
timestep = 0
loss = None


memory = Memory()

env = PathPlan(n_agents=NUM_AGENTS, n_anchor=NUM_IMU, file=directory)
print("Start position: ", env.start_locations)
print("End position: ", env.goal_locations)
RL = PPO(env, NUM_AGENTS)

for episode in tqdm(range(NUM_EPISODES)):
    cur_num_agents = NUM_AGENTS
    RL.change_num_agents(cur_num_agents)
    curRawState = env.reset(cur_num_agents)
        
    curState = curRawState

    episodeReward  = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    agent_episode_reward = [0] * cur_num_agents
        
    for step in range(LEN_EPISODES):
        timestep += 1

        action = RL.policy_old.act(curState, memory, cur_num_agents)
        newRawState  = env.step(action)
        [states, reward, done, info] = newRawState
            
            
        if step == LEN_EPISODES - 1:
            done = True
            
        for agent_index in range(cur_num_agents):
            memory.rewards.append(float(reward[agent_index])) 
            memory.is_terminals.append(done)
                
        newState = states
            
        # check time termination condition
        if timestep % UPDATE_TIMESTEP == 0:
            RL.update(memory)
            memory.clear_memory()
            timestep = 0
            
        # record history
        for i in range(cur_num_agents):
            agent_episode_reward[i] += reward[i]
                
        episodeReward += sum(reward)

        # set current state for next step
        curState = newState
            
        if done:
            break
            
    # post episode
    # if (episode%100 == 0):
    #     env.render()

    reward_history.append(episodeReward)
        
    for i in range(cur_num_agents):
        agent_history_dict[i].append((agent_episode_reward[i]))
        
        
    RL.summaryWriter_addMetrics(episode, loss, reward_history, agent_history_dict, LEN_EPISODES)
    if episode % 50 == 0:
        RL.saveModel(directory+"/checkpoints")

    if episode % 100 == 0:
        RL.saveModel(directory+"/checkpoints", True, episode)
                
        
RL.saveModel(directory+"/checkpoints")


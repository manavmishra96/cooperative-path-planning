# PPO implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Categorical, Normal
# from torch.utils.tensorboard import SummaryWriter
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import dgl
import cv2
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from env import PathPlan
from constants import CONSTANTS

const = CONSTANTS()

NUM_AGENTS = const.NUM_AGENTS
NUM_IMU = const.NUM_IMU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.stack = None
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

        
def generate_graph(num_nodes):
    env = PathPlan(n_agents = NUM_AGENTS, n_anchor = NUM_IMU, max_speed=5, goal_tolerance = 2)
    graph = dgl.DGLGraph()
    
    graph.add_nodes(num_nodes)
    edge_dict = env.generate_graph_()
    
    for i in range(num_nodes):
        if (len(edge_dict[i]) != 0):
            u = [i] * len(edge_dict[i])
            v = edge_dict[i]
            graph.add_edges(u,v)
    return graph  
    


class ActorCritic(nn.Module):
    def __init__(self, env, init_num_agents):
        super(ActorCritic, self).__init__()
        self.num_agents = init_num_agents 
        
        self.graph = generate_graph(init_num_agents).to(device)
        
        self.reg1 = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        
        self.reg2 = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        self.train()

    def change_num_agents(self, num_agents):
        self.graph = self.generate_graph(num_agents).to(device)
        return

    def generate_graph(self, num_nodes):
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)
        fc_src = [i for i in range(num_nodes)]
        for i in range(num_nodes):
            node_i = [i] * num_nodes
            graph.add_edges(fc_src, node_i)
        return graph
            
    def action_layer(self, x):
        x = self.reg1(x)
        return x
    
    def value_layer(self, x):
        x = self.reg2(x)
        return x
    
    def evaluate(self, state, action):
        action_param = self.action_layer(state)

        mu, sigma = action_param[:, :2], torch.exp(action_param[:, 2:])
        dist = Normal(mu, sigma)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def make_np(self, x):
        y = [o.data.cpu().numpy() for o in x]
        return y

    def act(self, state, memory, num_agents):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            action_param = self.action_layer(state)

            mu, sigma = action_param[:, :2], torch.exp(action_param[:, 2:])
            dist = Normal(mu, sigma)
            action = dist.sample()
                
            action_list = []
            for agent_index in range(num_agents):
                memory.states.append(state[agent_index])
                memory.actions.append(action[agent_index].view(-1))
                memory.logprobs.append(dist.log_prob(action[agent_index])[agent_index])
                action_list.append(action[agent_index])
        action_list = self.make_np(action_list)
        return action_list

    def act_max(self, state, memory, num_agents):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(device)
            action_param = self.action_layer(state)

            mu, sigma = action_param[:, :2], torch.exp(action_param[:, 2:])
            dist = Normal(mu, sigma)
            action = torch.argmax(action_param, dim=1)
                
            action_list = []
            for agent_index in range(num_agents):
                memory.states.append(state[agent_index])
                memory.actions.append(action[agent_index].view(1))
                memory.logprobs.append(dist.log_prob(action[agent_index])[agent_index])
                action_list.append(action[agent_index].item())
        return action_list

        
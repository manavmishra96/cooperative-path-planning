#PPO.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from statistics import mean


import matplotlib.pyplot as plt
import numpy as np
import os
import time
import dgl
import cv2
import networkx as nx

from actorcritic import ActorCritic, Memory
from constants import CONSTANTS

const = CONSTANTS()


import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Combining the ConvNet, GraphNet, and ActorCritic to perform the policy gradient optimization     
    
class PPO:
    def __init__(self, env, num_agents):
        self.lr = 0.000002
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4

        self.stack = []
        
        self.num_agents = num_agents
        
        torch.manual_seed(11)
        
        self.policy = ActorCritic(env, self.num_agents).to(device)
        # self.loadModel(filePath='checkpoints/ActorCritic_5600.pt')
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        # Needed for the clipped objective function
        self.policy_old = ActorCritic(env, self.num_agents).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        idx = 0
        while os.path.exists(f"tf_log/demo_CNN%s" % idx):
            idx = idx + 1   
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_CNN%s" % idx)
        print(f"Log Dir: {self.sw.log_dir}")
        
    def change_num_agents(self, num_agents):
        self.num_agents = num_agents
        self.policy.change_num_agents(num_agents)
        self.policy_old.change_num_agents(num_agents)
        
    def update(self, memory):
        all_rewards = []
        discounted_reward_list = [0] * int(self.num_agents)
        agent_index_list = list(range(self.num_agents)) * int(len(memory.rewards)/self.num_agents)
        for reward, is_terminal, agent_index in zip(reversed(memory.rewards), reversed(memory.is_terminals), reversed(agent_index_list)):
            if is_terminal:
                discounted_reward_list[agent_index] = 0
            discounted_reward_list[agent_index] = reward + (self.gamma * discounted_reward_list[agent_index])
            all_rewards.insert(0, discounted_reward_list[agent_index])

        all_rewards = torch.tensor(all_rewards).to(device)
        all_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        
        minibatch_sz = self.num_agents * const.len_episode
        
            
        mem_sz = len(memory.states)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            prev = 0
            for i in range(minibatch_sz, mem_sz+1, minibatch_sz):
                mini_old_states = memory.states[prev:i]
                mini_old_actions = memory.actions[prev:i]
                mini_old_logprobs = memory.logprobs[prev:i]
                mini_rewards = all_rewards[prev:i]
                
                # Convert list to tensor
                old_states = torch.stack(mini_old_states).to(device).detach()
                old_actions = torch.stack(mini_old_actions).to(device).detach()
                old_logprobs = torch.stack(mini_old_logprobs).to(device).detach()

                rewards = mini_rewards #torch.from_numpy(mini_rewards).float().to(device)
                
                prev = i
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())
                    
                # Finding Surrogate Loss:
                advantages = (rewards - state_values.detach()).view(-1,1)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy.mean()

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return 
    
    def formatInput(self, states):
        out = []
        for i in range(len(states[2])):
            temp = [states[2][i],states[3][i]]
            out.append(temp)
        return np.array(out)
    
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, loss, rewardHistory, agent_RwdDict, lenEpisode):
        if loss:
            self.sw.add_scalar('6.att_entropy', loss, episode)
        self.sw.add_scalar('3.Reward', rewardHistory[-1], episode)
        self.sw.add_scalar('5.Episode Length', lenEpisode, episode)
        
        if len(rewardHistory)>=100:
            avg_reward = mean(rewardHistory[-100:])
        else:    
            avg_reward = mean(rewardHistory) 
        self.sw.add_scalar('1.Average of Last 100 episodes', avg_reward, episode)
        
        for item in agent_RwdDict:
            title ='4.Agent ' + str(item+1)
            if len(agent_RwdDict[item]) >= 100:
                avg_agent_rwd=  agent_RwdDict[item][-100:]
            else:
                avg_agent_rwd =  agent_RwdDict[item]
            avg_agent_rwd = mean(avg_agent_rwd)

            self.sw.add_scalar(title,avg_agent_rwd, len(agent_RwdDict[item])-1)
            
    def summaryWriter_close(self):
        self.sw.close()
        
    def saveModel(self, filePath, per_save=False, episode=0):
        if per_save == False:
            torch.save(self.policy.state_dict(), f"{filePath}/{self.policy.__class__.__name__}.pt")
        else:
            torch.save(self.policy.state_dict(), f"{filePath}/{self.policy.__class__.__name__}_{episode}.pt")
    
    def loadModel(self, filePath, cpu = 0):
        if cpu == 1:
            self.policy.load_state_dict(torch.load(filePath, map_location=torch.device('cpu')))
        else:
            self.policy.load_state_dict(torch.load(filePath))
        self.policy.eval()
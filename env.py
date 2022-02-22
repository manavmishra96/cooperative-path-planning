## GYM ENVIRONMENT

# Torch Lib
from typing import ClassVar
from sklearn import neighbors
import torch 
import time
import os
import random
import math
import cv2
import glob

from torch._C import device 
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Gym Libs 
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# Python Libs 
import gym 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import seaborn as sns
import cv2 as cv
from sklearn.model_selection import train_test_split 
from itertools import cycle
from tqdm import tqdm

from constants import CONSTANTS
from obstacle import Obstacle

const = CONSTANTS()
obsmap = Obstacle()

class agent():
    """
    Class to define an agent and maitiain its properties
    """
    def __init__(self, x=np.random.randint(24, 34), y=np.random.randint(24, 34), imu=False, agent_id=0):
        self.x = x
        self.y = y

        # self.theta = np.random.uniform(-np.pi, np.pi)

        self.isIMU = imu
        self.id = agent_id
        self.is_localized = 0

        self.neighbors = []
        self.sorted_neig = []
        
        self.mu_t = np.matrix([[self.x],
                              [self.y]])
        self.sig_t = np.matrix([[1e-4, 0.0],
                               [0.0, 1e-4]])
        
        self.update_memory()
        self.R_t = const.action_uncertainity
        self.get_obs()
        

    def get_obs(self, z_t=None, Xg=None, Q_t=None, C_t=None, got_obs=False):
        """
        Updates the observation of the agent
        """
        self.z_t = z_t
        self.Xg = Xg
        self.Q_t = Q_t
        self.C_t = C_t
        self.got_obs = got_obs
        

    def update_memory(self):
        self.mu_t_mem = self.mu_t
        self.sig_t_mem = self.sig_t
        

    def update_agent(self, dx, dy, grid):
        self.update_memory()
        u_t = np.matrix([[dx],
                        [dy]])
        
        if (self.got_obs == False):
            self.mu_t, self.sig_t = self.kalman_filter(mu_t_=self.mu_t_mem, sig_t_=self.sig_t_mem, u_t=u_t, R_t=self.R_t, got_obs=False)
            
        else:
            self.mu_t, self.sig_t = self.kalman_filter(mu_t_=self.mu_t_mem, sig_t_=self.sig_t_mem, u_t=u_t, z_t=self.z_t, R_t=self.R_t, C_t=self.C_t, Q_t=self.Q_t, got_obs=True)
        
        self.mu_t = np.matrix([[self.mu_t.item(0,0)],
                              [self.mu_t.item(1,0)]])
        
        temp_loc = np.array([self.mu_t.item(0,0), self.mu_t.item(1,0)])
        
        if grid[int(temp_loc[0])][int(temp_loc[1])]:
            self.mu_t = self.mu_t_mem
            
        self.x = self.mu_t.item(0,0)
        self.y = self.mu_t.item(1,0)
        
        if(self.is_localized == True):
            self.sig_t = np.matrix([[1e-4, 0.0],
                                    [0.0, 1e-4]])
        

    def update_agent_imu(self, x, y):
        self.update_memory()
        self.x = x
        self.y = y
        self.mu_t = np.matrix([[x],[y]])

    
    def kalman_filter(self, mu_t_=None, sig_t_=None, u_t=None, z_t=None, 
        A_t=np.eye(2), B_t=np.eye(2), R_t=None, 
        C_t=None, Q_t=None, got_obs=False):

        """
        kalman filter algorithm as described in the paper
        """
        if (got_obs == False):
            mu_bar_t = A_t@mu_t_ + B_t@u_t
            sig_bar_t = A_t@sig_t_@A_t.T + R_t
            
            return mu_bar_t, sig_bar_t
        
        else:
            mu_bar_t = A_t@mu_t_ + B_t@u_t
            sig_bar_t = A_t@sig_t_@A_t.T + R_t

            K_t = sig_bar_t@C_t.T@np.linalg.pinv(C_t@sig_bar_t@C_t.T + Q_t)
            
            mu_t = mu_bar_t + K_t@(z_t - C_t@mu_bar_t)
            sig_t = (np.eye(sig_t_.shape[0]) - K_t@C_t)@sig_bar_t
            
            return mu_t, sig_t


########################################################################################################################################################
    

class PathPlan(gym.Env):
    def __init__(self,
                n_agents = 2,
                n_anchor = 2,
                file=None, 
                max_speed = 4.0,
                goal_tolerance = 2.0,
                decay = 1,
                Rmin = -400
                ):
        super(PathPlan, self).__init__()
        
        self.steps = 0
        self.length = const.gridSize
        self.episodes = 0
        self.decay = decay
        self.Rmin = Rmin
        self.file = file

        self.grid, self.vsbs = obsmap.getObstacleMap(np.zeros((const.gridSize, const.gridSize)), obsmap.open_map())
        
        self.n_agents = n_agents
        self.n_anchor = n_anchor

        self.max_speed = max_speed
        self.goal_tolerance = goal_tolerance
        self.rho = 10
        # self.num_beams = 16
        
        
        # self.observation_space = spaces.Box(-self.length, self.length)
        self.color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        """Randomly initializing the agents"""
        self.start_locations = np.random.randint(0 + 5, self.length - 5 , size=(self.n_agents, 2))
        self.goal_locations = np.random.randint(0 + 5, self.length - 5, size=(self.n_agents, 2))

        """Define the agents from Agent class"""
        self.agents = [agent(x=self.start_locations[i][0], y=self.start_locations[i][1], imu=False, agent_id=i) for i in range(self.n_agents)]

        if self.n_agents == self.n_anchor:
            anchor_idx = range(n_agents)
        else:
            anchor_idx = random.sample(range(1, self.n_agents), self.n_anchor)

        for idx in anchor_idx:
            self.agents[idx].isIMU = True
            self.agents[idx].is_localized = 1

        self.graph = self.generate_graph_()
        

    def reset(self, n_agents):
        self.steps = 0
        self.episodes += 1

        self.state = []
        self.reward = [0] * self.n_agents
        self.done = [False] * self.n_agents
        self.info = [{}] * self.n_agents


        """Define the agents from Agent class"""
        self.agents = [agent(x=self.start_locations[i][0], y=self.start_locations[i][1], imu=False, agent_id=i) for i in range(self.n_agents)]

        if self.n_agents == self.n_anchor:
            anchor_idx = range(n_agents)
        else:
            anchor_idx = random.sample(range(1, self.n_agents), self.n_anchor)

        for idx in anchor_idx:
            self.agents[idx].isIMU = True
            self.agents[idx].is_localized = 1

        for i, agt in enumerate(self.agents):
            self.state.append([agt.x, agt.y, agt.is_localized])

        self.graph = self.generate_graph_()
        return np.array(self.state)


    def step(self, action):
        self.steps += 1
        delta_t = 0.1
        old_state = self.state.copy()
        new_state = []

        for i, act in enumerate(action):
            vx = self.max_speed * act[0]
            vy = self.max_speed * act[1]
            # print(vx,vy)

            dx, dy = vx * delta_t, vy * delta_t

            temp_dx, temp_dy = self.agents[i].x + dx, self.agents[i].y + dy

            if not self.obs_collison(temp_dx,temp_dy):
                self.agents[i].x = self.agents[i].x + dx
                self.agents[i].y = self.agents[i].y + dy
            else:
                self.agents[i].x = self.agents[i].x 
                self.agents[i].y = self.agents[i].y 

            self.graph = self.generate_graph_()
            self.update_agent_obs(self.agents[i]) 

            if (self.agents[i].isIMU == True):
                self.agents[i].update_agent_imu(self.agents[i].x, self.agents[i].y)
            else:
                self.agents[i].update_agent(dx, dy, self.grid)

            location = np.array([self.agents[i].x, self.agents[i].y])
            self.reward[i] = -0.5 * np.linalg.norm(self.goal_locations[i] - location) #+ 0.5 * 50 * self.agents[i].is_localized
            self.done[i] = False
            self.info[i] = {}

            new_state.append([self.agents[i].x, self.agents[i].y, self.agents[i].is_localized])

            if np.linalg.norm(self.goal_locations[i] - location) < self.goal_tolerance:
                self.reward[i] = 200
                self.done[i] = True

        self.state = new_state
            
        if (self.episodes%1000 == 0): 
            self.plot_map()
            if (self.steps == const.len_episode):
                self.render()
            

        # self.observation = self._get_observation(self.state)
        return np.array(self.state), self.reward, all(self.done), self.info
    

    def obs_collison(self, dx, dy):
        dx, dy = min(dx, const.gridSize-1), min(dy, const.gridSize-1)
        return True if (self.grid[int(dx)][int(dy)] == 200) else False


    def generate_graph_(self):
        g = {}
        for agent in self.agents:
            agent.neighbors = []
            for agent_ in self.agents:
                if (agent == agent_):
                    continue
                else:
                    if (np.linalg.norm(np.array([agent.x, agent.y]) - np.array([agent_.x, agent_.y])) < self.rho):
                        agent.neighbors.append(agent_.id)
            g[agent.id] = agent.neighbors
        agent.sorted_neig = sorted(agent.neighbors, key = lambda p: math.sqrt((self.agents[p].x - agent.x)**2 + (self.agents[p].y - agent.y)**2))
        return g


    def update_agent_obs(self, agt):
        local_check = [self.agents[neig].is_localized for neig in agt.neighbors]
        sorted_neighbors = sorted(agt.neighbors, key = lambda p: math.sqrt((self.agents[p].x - agt.x)**2 + (self.agents[p].y - agt.y)**2))
        
        if(agt.isIMU == False and any(local_check) == 1 and len(sorted_neighbors) > 0):
            agt.is_localized = 1
            n_neighbor = sorted_neighbors[0]
            
            x, y = agt.x, agt.y
            xg, yg = self.agents[n_neighbor].x, self.agents[n_neighbor].y

            Xg = np.matrix([[xg], [yg]])

            a, b = xg - x, yg - y
            z_t = np.matrix([[a], [b]])

            C_t = np.matrix([[a/(xg-a + 1e-4), 0.0],
                            [0.0, b/(yg-b + 1e-4)]])
                
            Q_t = self.agents[n_neighbor].sig_t if (agt.is_localized == 0) else const.meas_uncertainity
            agt.get_obs(z_t=z_t, Xg=Xg, Q_t=Q_t, C_t=C_t, got_obs=True)
                
        else:
            agt.get_obs(got_obs=False)


    def plot_map(self):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        # plt.title("RL-DCL simulations")
        plt.xlim([0, const.gridSize])
        plt.ylim([0, const.gridSize])

        cmap = plt.cm.binary
        plt.imshow(self.grid, cmap=cmap, vmin=0, vmax=200, alpha=0.85)

        for i, agent in enumerate(self.agents):
            if agent.isIMU:
                plt.plot(agent.x, agent.y, marker='s', color=self.color[i])
            else:
                plt.plot(agent.x, agent.y, marker='o', color=self.color[i])
            plt.scatter(self.goal_locations[i][0], self.goal_locations[i][1], s=250, facecolors='none', edgecolors=self.color[i], linestyle='--')

        plt.savefig(self.file+"/images/%s.jpg" % str(self.steps), bbox_inches='tight',pad_inches = 0)
        plt.close()
        return
        
    def render(self):
        img_array = self.load_images_from_folder(self.file+'/images/')
        height, width, layers = img_array[0].shape
        size = (width,height)

        out = cv2.VideoWriter(self.file+'/renders/%s.mp4' % str(self.episodes), cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        
        out.release()
        
        filelist = glob.glob(os.path.join(self.file+"/images/", "*.jpg"))
        for f in filelist:
            os.remove(f)

        del img_array
           

    def load_images_from_folder(self, folder):
        images = []
        for filename in self.sorted_alphanumeric(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images
    

    def sorted_alphanumeric(self, data):
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)



if __name__=='__main__':
    env = PathPlan(n_agents=3, n_anchor=1)
    print(env.grid)
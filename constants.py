#constants.py

import numpy as np
import os
import json

class CONSTANTS:
    def __init__(self):
        self.epsilon = 0.00001
        self.gridSize = 50
        self.commRange = 1000
        self.decay = 1
        self.Rmin = -400
        self.num_episode = 30000
        self.len_episode = 1000

        self.NUM_AGENTS = 1
        self.NUM_IMU = 1

        self.directory = "models/test5"
        
        self.isShared = True
        self.action_uncertainity = np.matrix([[0.5, 0.0],
                                             [0.0, 0.5]])
        self.meas_uncertainity = np.matrix([[1e-4, 0.0],
                                           [0.0, 1e-4]])

        self.GRID_CENTER_PTS = self.getGridCenterPts()

    def getGridCenterPts(self):
        x, y = np.meshgrid(np.arange(self.gridSize), np.arange(self.gridSize))
        x, y = x.flatten()+0.5, y.flatten() + 0.5
        points = np.vstack((x,y)).T
        return points
import skgeom as sg
import numpy as np
from constants import CONSTANTS as K
from matplotlib.path import Path
CONST = K()

from visibility import Visibility

class Obstacle:
    def __init__(self):
        pass

    def getObstacleMap(self, emptyMap, obstacleSet):
        obsList = obstacleSet
        vsb  = Visibility(emptyMap.shape[0], emptyMap.shape[1])
        for obs, isHole in obsList:
            vsb.addGeom2Arrangement(obs)
        
        isHoles = [obs[1] for obs in obsList]
        if any(isHoles) == True:
            pass
        else:
            vsb.boundary2Arrangement(vsb.length, vsb.height)
        
        # get obstacle polygon
        points = CONST.GRID_CENTER_PTS
        img = np.zeros_like(emptyMap, dtype = bool)
        for obs, isHole in obsList:
            p = Path(obs)
            grid = p.contains_points(points)
            mask = grid.reshape(CONST.gridSize, CONST.gridSize)
            img = np.logical_or(img , (mask if not isHole else np.logical_not(mask)))
           
        img = img.T
        img = np.where(img, 200, emptyMap)
        return img, vsb

    def map_4_room(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[15,30],
                [15,35],
                [12,35],
                [12,45],
                [21,45],
                [21,35],
                [18,35],
                [18,30],
                [27,30],
                [27,35],
                [24,35],
                [24,45],
                [33,45],
                [33,35],
                [30,35],
                [30,30],
                [30,20],
                [30,15],
                [33,15],
                [33,5],
                [24,5],
                [24,15],
                [27,15],
                [27,20],
                [18,20],
                [18,15],
                [21,15],
                [21,5],
                [12,5],#
                [12,15],
                [15,15],
                [15,20]]
        obsList.append([geom, isHole])
        
        return obsList
    
    #2-room map
    def map_2_room(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[20,4],
                [20,16],
                [28,16],
                [28,25],
                [28,34],
                [20,34],
                [20,46],
                [40,46],
                [40,34],
                [32,34],
                [32,25],
                [32,16],
                [40,16],
                [40,4]]
        obsList.append([geom, isHole])
        
        return obsList

    #open map
    def open_map(self):
        obsList = []
        # add points in CW order and 
        isHole = True
        geom = [[5,5],
                [5, 45],
                [45,45],
                [45,5]]
        obsList.append([geom, isHole])
        return obsList
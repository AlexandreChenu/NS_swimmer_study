#!/usr/bin python -w

import datetime
import os
from os import path
import array
import time
import random
import pickle
import copy
import argparse

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D

import gym
import pandas as pd
import numpy as np

import gym

from numpy import load


def plot_lineage(bds):

    fig, ax = plt.subplots()
    # ax.set_xlim(-0.1, 0.1)
    # ax.set_ylim(-0.1, 0.1)
    L_bd_X = [bd[0] for bd in bds]
    L_bd_Y = [bd[1] for bd in bds]

    L_dx = []
    L_dy = []
    for i in range(1,len(bds)-1):
        dx = bds[i+1][0]-bds[i][0]
        dy = bds[i+1][1]-bds[i][1]
        arrow = plt.arrow(bds[i][0], bds[i][1], dx, dy, alpha=0.6, width = 0.03)
        ax.add_patch(arrow)

    plt.scatter(L_bd_X, L_bd_Y, c="red")


    plt.show()
    return

with open('/Users/chenu/Desktop/PhD/github/trajectory_AE/diversity_algorithms/diversity_algorithms/experiments/SwimmerDet_NS_2022_01_17-15:57:30_0/lineages.pkl', 'rb') as handle:
    lineages = pickle.load(handle)
with open('/Users/chenu/Desktop/PhD/github/trajectory_AE/diversity_algorithms/diversity_algorithms/experiments/SwimmerDet_NS_2022_01_17-15:57:30_0/id_to_bd.pkl', 'rb') as handle:
    id_to_bd = pickle.load(handle)

last_id = list(id_to_bd.keys())[-1]
print("last_id = ", last_id)
last_id_bd = id_to_bd[last_id]
print("last_id_bd = ", last_id_bd)


id = last_id
path = []
bd = last_id_bd
while id is not None:
    bd = id_to_bd[id]
    path = [bd] + path
    id = lineages[id]
    print("bd = ", bd)

plot_lineage(path)

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

from diversity_algorithms.controllers import SimpleNeuralController

import gym
import pandas as pd
import numpy as np

import gym

from numpy import load


def plot_trajectory(traj):

    plt.scatter(traj[-1][0], traj[-1][1], c="r")

    L_X = [state[0] for state in traj]
    L_Y = [state[1] for state in traj]
    plt.plot(L_X, L_Y, alpha = 0.5, c = "b")
    # plt.xlim((-1.6, 1.6))
    plt.ylim((-1.5,2.))
    return

data = load('/Users/chenu/Desktop/PhD/github/trajectory_AE/diversity_algorithms/diversity_algorithms/experiments/SwimmerTB_NS_2021_12_15-12:12:08_0/population_traj_gen4990.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

# ind = data["ind_100"]

plt.figure()
for i in range(0,100):
    traj = data["traj_"+str(i)]
    plot_trajectory(traj)

# indx = np.random.randint(0,100)
# traj = data["traj_"+str(indx)]
# plot_trajectory(traj)
plt.show()

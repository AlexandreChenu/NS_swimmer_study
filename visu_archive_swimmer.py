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


def plot_pop_bd(bd_pop):

    fig, ax = plt.subplots()
    # ax.set_xlim(-0.1, 0.1)
    # ax.set_ylim(-0.1, 0.1)
    L_bd_X = [bd[0] for bd in bd_pop]
    L_bd_Y = [bd[1] for bd in bd_pop]

    plt.scatter(L_bd_X, L_bd_Y, c="red")

    plt.show()
    return


data = load('/Users/chenu/Desktop/PhD/github/trajectory_AE/diversity_algorithms/diversity_algorithms/experiments/SwimmerDet_NS_2022_01_26-19:23:49_0/bd_population_bd_gen300.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

bd_pop = []
for i in range(0,100):
    bd_pop.append(data["bd_"+str(i)])


# bd_pop = []
# for i in range(0,len(lst)-2, 5):
#     bd_pop.append(data["bd_"+str(i)])

plot_pop_bd(bd_pop)

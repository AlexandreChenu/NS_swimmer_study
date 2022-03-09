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

from diversity_algorithms.environments import *

from diversity_algorithms.controllers import SimpleNeuralController

import gym
import pandas as pd
import numpy as np

import gym

from numpy import load


def run_ind(ind_vec):

    env = gym.make('SwimmerDet-v3')
    new_state = env.reset()

    model = SimpleNeuralController(np.concatenate([env.sim.data.qpos.flat, env.sim.data.qvel.flat]).shape[0], env.action_space.sample().shape[0], n_hidden_layers=2, n_neurons_per_hidden=10)
    model.set_parameters(ind_vec)



    R = 0

    for t in range(0,300):
        env.render()

        obs = np.concatenate([env.sim.data.qpos.flat, env.sim.data.qvel.flat])
        new_action = model.predict(obs)
        # print("new_action = ", new_action)
        new_state, reward,_,info = env.step(new_action)
        print("state = ", env.sim.data.qpos[:2])
        R+= reward

    print("reward = ", reward)

data = load('/Users/chenu/Desktop/PhD/github/trajectory_AE/diversity_algorithms/diversity_algorithms/experiments/SwimmerDet_NS_2022_01_17-14:32:57_0/final_pop_all_gen100.npz')
lst = data.files

lst = data.files
for item in lst:
    print(item)
    print(data[item])

# indx = np.random.randint(0,100)
# print("indx = ", indx)
indx = 1
ind = data["ind_"+str(indx)]

run_ind(ind)

print("bd indx = ", data["bd_"+str(indx)])

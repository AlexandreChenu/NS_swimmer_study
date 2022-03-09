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

import seaborn

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



def get_max_dist(dir):

    max_dist = 0.
    popsize = 100
    L_max_dist = [0.]
    for gen in range(100,600,100):
        print("gen = ", gen)
        filename = dir + "/bd_population_bd_gen" + str(gen) + ".npz"
        data = load(filename)

        bd_pop = []
        for i in range(0,popsize):
            bd = data["bd_"+str(i)]
            # print("bd = ", bd)
            dist_to_origin= np.linalg.norm(bd - np.array([0,0]))
            if dist_to_origin > max_dist:
                max_dist = dist_to_origin

        L_max_dist.append(max_dist)

        print("max_dist = ", max_dist)

    return L_max_dist

def get_stats(dir, ind_size, nb_gen, extension = ""):

    max_dists = np.zeros((19, 600//100))
    for i in range(1,20):
        local_dir = dir + "NS_swimmer" + str(ind_size) + "_" + str(i) + extension
        L_dir = [f for f in os.listdir(local_dir) if "Swimmer-ep" in f]
        full_path = local_dir+"/"+L_dir[0]
        L_max_dist = get_max_dist(full_path)

        for indx in range(len(L_max_dist)):
            max_dists[i-1][indx] = L_max_dist[indx]

    return np.mean(max_dists, axis=0), np.std(max_dists, axis=0)


dir = "/Users/chenu/Desktop/PhD/github/trajectory_AE/data/results/swimmer_10/res_10_200/"
m_dist_10, std_dist_10 = get_stats(dir, 10, 200, extension = "/res_10_200/")
print("m_dist_10 = ", m_dist_10)
print("std_dist_10 = ", std_dist_10)

dir = "/Users/chenu/Desktop/PhD/github/trajectory_AE/data/results/swimmer_good_10/res_good_10_200/"
m_dist_good_10, std_dist_good_10 = get_stats(dir, 10, 200)
print("m_dist_good_10 = ", m_dist_good_10)
print("std_dist_good_10 = ", std_dist_good_10)

dir = "/Users/chenu/Desktop/PhD/github/trajectory_AE/data/results/swimmer_50/res_50_200/"
m_dist_50, std_dist_50 = get_stats(dir, 50, 200)
print("m_dist_10 = ", m_dist_50)
print("std_dist_10 = ", std_dist_50)

dir = "/Users/chenu/Desktop/PhD/github/trajectory_AE/data/results/swimmer_good_50/res_good_50_200/"
m_dist_good_50, std_dist_good_50 = get_stats(dir, 50, 200)
print("m_dist_10 = ", m_dist_good_50)
print("std_dist_10 = ", std_dist_good_50)

seaborn.set()

plt.figure()

plt.title("max distance to origin")

# iters = np.linspace(0,len(list(m_dist_10)))
iters = [i for i in range(1,len(list(m_dist_10))+1)]
plt.plot(iters, m_dist_10, c="black", label="NN 10,10 (uniform init)")
plt.fill_between(iters, m_dist_10 - std_dist_10, m_dist_10  + std_dist_10 , color="dimgrey", alpha = 1.)

iters = np.linspace(0,len(list(m_dist_50)))
iters = [i for i in range(1,len(list(m_dist_50))+1)]
plt.plot(iters, m_dist_50, c="purple", label="NN 50,50 (uniform init)")
plt.fill_between(iters, m_dist_50 - std_dist_50, m_dist_50  + std_dist_50 , color="plum", alpha = 0.8)

iters = [i for i in range(1,len(list(m_dist_good_10))+1)]
plt.plot(iters, m_dist_good_10, c="green", label="NN 10,10 (normal init)")
plt.fill_between(iters, m_dist_good_10 - std_dist_good_10, m_dist_good_10  + std_dist_good_10 , color="seagreen", alpha = 0.6)

iters = [i for i in range(1,len(list(m_dist_good_50))+1)]
plt.plot(iters, m_dist_good_50, c="blue", label="NN 50,50 (normal init)")
plt.fill_between(iters, m_dist_good_50 - std_dist_good_50, m_dist_good_50  + std_dist_good_50 , color="skyblue", alpha = 0.4)

plt.xlabel("generation (x10)")
plt.ylabel("maximum distance to origin")

plt.legend()
plt.show()

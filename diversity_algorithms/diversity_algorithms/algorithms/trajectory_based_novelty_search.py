#!/usr/bin python -w

import random
from scipy.spatial import KDTree
import numpy as np
import datetime
import os
import array


creator = None
def set_creator(cr):
    global creator
    creator = cr

import pickle

import copy

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *

from diversity_algorithms.algorithms.trajectory_based_novelty_management import *

import alphashape
from shapely.geometry import Point, Polygon, LineString

import torch
from torch.optim import AdamW
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from .model import TrajectoryDecoder, TrajectoryEncoder, ActorMLP

__all__=["novelty_ea"]

import sys


def build_toolbox_ns(evaluate,params,pool=None):

    toolbox = base.Toolbox()

    if(params["geno_type"] == "realarray"):
        print("** Using fixed structure networks (MLP) parameterized by a real array **")
        # With fixed NN
        # -------------
        toolbox.register("attr_float", lambda : random.uniform(params["min"], params["max"]))

        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=params["ind_size"])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        #toolbox.register("mate", tools.cxBlend, alpha=params["alpha"])

        # Polynomial mutation with eta=15, and p=0.1 as for Leni
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=params["eta_m"], indpb=params["indpb"], low=params["min"], up=params["max"])
    else:
        raise RuntimeError("Unknown genotype type %s" % geno_type)

    #Common elements - selection and evaluation

    v=str(params["variant"])
    variant=v.replace(",","")
    if (variant == "NS"):
        toolbox.register("select", tools.selBest, fit_attr='novelty')
    elif (variant == "Fit"):
        toolbox.register("select", tools.selBest, fit_attr='fitness')
    elif (variant == "Random"):
        toolbox.register("select", random.sample)
    elif (variant == "DistExplArea"):
        toolbox.register("select", tools.selBest, fit_attr='dist_to_explored_area')
    else:
        print("Variant not among the authorized variants (NS, Fit, Random, DistExplArea), assuming multi-objective variant")
        toolbox.register("select", tools.selNSGA2)

    toolbox.register("evaluate", evaluate)

    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)


    return toolbox

def batch_generator(batch_size, traj_data, traj_mask_data):
    all_idx = torch.randperm(traj_data.shape[0])
    start = 0
    while start < traj_data.shape[0]:
        idx = all_idx[start: start+batch_size]
        start += batch_size
        yield traj_data[idx], traj_mask_data[idx]

def train_encoder(encoder, decoder, optimizer, trajs, masks):
    traj_data = torch.as_tensor(trajs, dtype=torch.float).to(device)
    traj_mask = torch.as_tensor(masks, dtype=torch.bool).to(device)

    for e in range(1):
        for batch in batch_generator(200, traj_data, traj_mask):
            traj_data_batch, traj_mask_batch = batch
            # print("traj_data_batch = ", traj_data_batch.shape)
            # print("traj_mask_batch = ", traj_mask_batch.shape)
            embedding = encoder(traj_data_batch, traj_mask_batch)
            reconstructed_traj = decoder(embedding, traj_mask_batch)

            loss = (traj_data_batch[~traj_mask_batch] - reconstructed_traj[~traj_mask_batch]) ** 2
            loss = loss.mean()
            print(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

## DEAP compatible algorithm
def novelty_ea(evaluate, params, pool=None):
    """Novelty Search algorithm

    Novelty Search algorithm. Parameters:
    :param evaluate: the evaluation function
    :param params: the dict of run parameters
       * params["pop_size"]: the number of parent individuals to keep from one generation to another
       * params["lambda"]: the number of offspring to generate, given as a multiplying coefficent on params["pop_size"] (the number of generated individuals is: params["lambda"]*params["pop_size"]
       * params["nb_gen"]: the number of generations to compute
       * params["stats"]: the statistics to use (from DEAP framework))
       * params["stats_offspring"]: the statistics to use (from DEAP framework))
       * params["variant"]: the different supported variants ("NS", "Fit", "NS+Fit", "NS+BDDistP", "NS+Fit+BDDistP"), add "," at the end of the variant to select in offspring only (as the ES "," variant). By default, selects within the parents and offspring. "NS" uses the novelty criterion, "Fit" the fitness criterion and "BDDistP' the distance to the parent in the behavior space. If a single criterion is used, an elitist selection scheme is used. If more than one criterion is used, NSGA-II is used (see build_toolbox_ns function)
       * params["cxpb"]: the crossover rate
       * params["mutpb"]: the mutation rate
    :param dump_period_bd: the period for dumping behavior descriptors
    :param dump_period_pop: the period for dumping the current population
    :param evolvability_period: period of the evolvability computation
    :param evolvability_nb_samples: the number of samples to generate from each individual in the population to estimate their evolvability (WARNING: it will significantly slow down a run and it is used only for statistical reasons
    """
    print("Novelty search algorithm")

    alphas=params["alphas"] # parameter to compute the alpha shape, to estimate the distance to explored area

    variant=params["variant"]
    if ("+" in variant):
        emo=True
    else:
        emo=False

    nb_eval=0

    lambda_ = int(params["lambda"]*params["pop_size"])

    toolbox=build_toolbox_ns(evaluate,params,pool)

    population = toolbox.population(n=params["pop_size"])

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals']

    if (params["stats"] is not None):
        logbook.header += params["stats"].fields
    if (params["stats_offspring"] is not None):
        logbook.header += params["stats_offspring"].fields

    # Load pre-trained transfoAE

    # Hyperparams
    obs_shape = (2,)
    d_model = 10
    nhead = 10
    dim_feedforward = 256
    dropout = 0
    activation = 'gelu'
    demo_max_length = 100
    batch_size = 128
    num_layers = 3
    lr = 0.002
    n_epoch = 20000
    min_val_loss = float('Inf')
    train_delta_iter = [n_train * 10 for n_train in range(1,20)]

    train_freqs=[train_delta_iter[0]]
    sum_delta_iter = train_delta_iter[0]
    for delta_iter in train_delta_iter[1:]:
        sum_delta_iter += delta_iter
        train_freqs.append(sum_delta_iter)
    print("train_freqs = ", train_freqs)

    dict_model = torch.load('/Users/chenu/Desktop/PhD/github/trajectory_AE/model_ae_simple_swimmer.pth')
    # dict_model = torch.load('/home/chenu/git/trajectory_AE/diversity_algorithms/diversity_algorithms/models/model_ae_simple_swimmer.pth')
    encoder = TrajectoryEncoder(obs_shape, d_model, nhead, dim_feedforward, dropout, activation, demo_max_length, num_layers)
    encoder.load_state_dict(dict_model["encoder_params"])
    decoder = TrajectoryDecoder(obs_shape, d_model, nhead, dim_feedforward, dropout, activation, demo_max_length, num_layers)
    decoder.load_state_dict(dict_model["decoder_params"])
    model_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(model_params, lr=0.0005, weight_decay=1e-3)

    encoder.to(device)
    decoder.to(device)

    print("encoder.device = ", device)

    trajs = []
    masks = []

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    nb_eval+=len(invalid_ind)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # fit is a list of fitness (that is also a list) and behavior descriptor

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fit = fit[0] # fit is an attribute just used to store the fitness value
        ind.parent_bd=None

        ## compute bd using transformer
        traj = listify(fit[1])
        trajs.append(traj)
        mask = listify(fit[2])
        masks.append(mask)

        ind.traj = copy.deepcopy(traj)
        ind.mask = copy.deepcopy(mask)
        ind.bd = None

        # t_traj = torch.as_tensor([traj], dtype=torch.float)
        # t_mask = torch.as_tensor([mask], dtype=torch.bool)
        # bd = encoder(t_traj, t_mask).detach().numpy()[0]
        # # print("bd = ", bd)
        # ind.bd= bd

        ind.id = generate_uuid()
        ind.parent_id = None

    for ind in population:
        ind.am_parent=0

    total_individuals = copy.deepcopy(population)

    archive=updateNovelty(population,population,None,encoder,params)

    isortednov=sorted(range(len(population)), key=lambda k: population[k].novelty, reverse=True)


    for i,ind in enumerate(population):
        ind.rank_novelty=isortednov.index(i)
        ind.fitness.values=ind.fit
        # if it is not a multi-objective experiment, the select tool from DEAP
        # has been configured above to take the right attribute into account
        # and the fitness.values is thus ignored
    gen=0

    # Do we look at the evolvability of individuals (WARNING: it will make runs much longer !)
    # generate_evolvability_samples(params, population, gen, toolbox)

    # record = params["stats"].compile(population) if params["stats"] is not None else {}
    # record_offspring = params["stats_offspring"].compile(population) if params["stats_offspring"] is not None else {}
    # logbook.record(gen=0, nevals=len(invalid_ind), **record, **record_offspring)
    # if (verbosity(params)):
    #     print(logbook.stream)

    #generate_dumps(params, population, None, gen, pop1label="population", archive=None, logbook=None)

    # Begin the generational process
    for gen in range(1, params["nb_gen"] + 1):

        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, params["cxpb"], params["mutpb"])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        nb_eval+=len(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fit = fit[0]
            ind.fitness.values = fit[0]
            ind.parent_bd=ind.bd
            ind.parent_id=ind.id
            ind.id = generate_uuid()

            # ind.bd=listify(fit[1])
            ## compute bd using transformer
            traj = listify(fit[1])
            trajs.append(traj)
            mask = listify(fit[2])
            masks.append(mask)

            ind.traj = copy.deepcopy(traj)
            ind.mask = copy.deepcopy(mask)
            ind.bd = None

            # t_traj = torch.as_tensor([traj], dtype=torch.float)
            # t_mask = torch.as_tensor([mask], dtype=torch.bool)
            # bd = encoder(t_traj, t_mask).detach().numpy()[0]
            # ind.bd= bd

        for ind in population:
            ind.am_parent=1
        for ind in offspring:
            ind.am_parent=0

        pq=population+offspring

        total_individuals += copy.deepcopy(offspring)

        pop_for_novelty_estimation=list(pq)
        archive=updateNovelty(pq,offspring,archive,encoder,params, pop_for_novelty_estimation)

        isortednov=sorted(range(len(pq)), key=lambda k: pq[k].novelty, reverse=True)

        for i,ind in enumerate(pq):
            ind.fitness.values=ind.fit

        if (verbosity(params)):
            print("Gen %d"%(gen))
        else:
            if(gen%100==0):
                print(" %d "%(gen), end='', flush=True)
            elif(gen%10==0):
                print("+", end='', flush=True)
            else:
                print(".", end='', flush=True)


        # Select the next generation population
        if ("," in variant):
            population[:] = toolbox.select(offspring, params["pop_size"])
        else:
            population[:] = toolbox.select(pq, params["pop_size"])

        if (("eval_budget" in params.keys()) and (params["eval_budget"]!=-1) and (nb_eval>=params["eval_budget"])):
            params["nb_gen"]=gen
            terminates=True
        else:
            terminates=False

        # Re-train transformer with new trajectories
        if(gen in train_freqs):
            train_encoder(encoder, decoder, optimizer, trajs, masks)
            # Update archive
            archive.update_bd_to_new_encoder(encoder)

        if(gen%10==0):
            #dump_data(population, gen, params, prefix="population", attrs=["all", "dist_to_explored_area", "dist_to_parent", "rank_novelty"], force=terminates)
            dump_data(population, gen, params, prefix="population", attrs=["traj"], force=terminates)
            # dump_data(total_individuals, gen, params, prefix="total_individuals", attrs=["traj"], force=terminates)

            dump_data(population, gen, params, prefix="bd", complementary_name="population", attrs=["bd"], force=terminates)
            # dump_data(total_individuals, gen, params, prefix="bd", complementary_name="total_individuals", attrs=["bd"], force=terminates)
            dump_data(offspring, gen, params, prefix="bd", complementary_name="offspring", attrs=["bd"], force=terminates)
            dump_data(archive.get_content_as_list(), gen, params, prefix="archive", attrs=["all"], force=terminates)

        generate_evolvability_samples(params, population, gen, toolbox)

        # Update the statistics with the new population
        # record = params["stats"].compile(population) if params["stats"] is not None else {}
        # record_offspring = params["stats_offspring"].compile(offspring) if params["stats_offspring"] is not None else {}
        # logbook.record(gen=gen, nevals=len(invalid_ind), **record, **record_offspring)
        # if (verbosity(params)):
        #     print(logbook.stream)


        if (terminates):
            break


    return population, archive, logbook, nb_eval





if (__name__=='__main__'):
    print("Test of the Novelty-based ES")

    printf("TODO...")

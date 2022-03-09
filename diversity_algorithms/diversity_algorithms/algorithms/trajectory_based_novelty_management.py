
import random
from scipy.spatial import cKDTree as KDTree
import numpy as np
from operator import attrgetter
from diversity_algorithms.algorithms.utils import verbosity
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

__all__ = ["NovArchive", "updateNovelty"]

# ### Novelty-based Evolution Strategies

class NovArchive:
    """Archive used to compute novelty scores."""
    def __init__(self, trajs, masks, encoder, k=15):
        self.all_traj=trajs
        self.all_mask=masks
        self.all_bd=[encoder(torch.as_tensor([traj], dtype=torch.float).to(device), torch.as_tensor([mask], dtype=torch.bool).to(device)).detach().cpu().numpy()[0] for traj,mask in zip(self.all_traj, self.all_mask)]

        if (len(self.all_bd)>0):
            # print("self.all_bd = ", self.all_bd)
            self.kdtree=KDTree(self.all_bd)
        else:
            self.kdtree=None # for archive-less experiments
        self.k=k
        #print("Archive constructor. size = %d"%(len(self.all_bd)))

    def ready(self):
        return self.size()>self.k

    def get_content_as_list(self):
        return self.all_bd

    def update(self,new_traj, new_mask, encoder):
        if (len(new_traj)==0):
            return
        oldsize=len(self.all_traj)
        if (oldsize>0):
            for traj in new_traj:
                assert len(traj)==len(self.all_traj[0]), "update archive, traj of different sizes: len(traj)=%d, len(all_traj[0])=%d"%(len(traj),len(all_traj[0]))

        self.all_bd=self.all_bd + [encoder(torch.as_tensor([traj], dtype=torch.float).to(device), torch.as_tensor([mask], dtype=torch.bool).to(device)).detach().cpu().numpy()[0] for traj,mask in zip(new_traj, new_mask)]
        self.kdtree=KDTree(self.all_bd)
        #print("Archive updated, old size = %d, new size = %d"%(oldsize,len(self.all_bd)))

    def update_bd_to_new_encoder(self, encoder):
        ## compute new bd with saved trajectories
        self.all_bd=[encoder(torch.as_tensor([traj], dtype=torch.float).to(device), torch.as_tensor([mask], dtype=torch.bool).to(device)).detach().cpu().numpy()[0] for traj,mask in zip(self.all_traj, self.all_mask)]
        self.kdtree=KDTree(self.all_bd) ## Update kdtree

    def get_nov(self, bd, population=[]):
        if(True in np.isnan(bd)):
            return -1

        #if (len(population)==0):
        #    print("WARNING: get_nov with an empty population")
        dpop=[]

        for ind in population:
            if (True in np.isnan(ind.bd)):
                continue
            assert len(bd)==len(ind.bd), "get_nov, bd of different sizes: len(bd)=%d, len(ind.bd)=%d"%(len(bd),len(ind.bd))

            dpop.append(np.linalg.norm(bd-ind.bd))

        if (self.kdtree is None):
            darch=[] # archive-less NS (i.e. behavior diversity)
        else:
            darch,ind=self.kdtree.query(bd,self.k, n_jobs=-1)
        d=dpop+list(darch)
        d.sort()
        #if (d[0]!=0):
        #    print("WARNING in novelty search: the smallest distance should be 0 (distance to itself). If you see it, you probably try to get the novelty with respect to a population your indiv is not in. The novelty value is then the sum of the distance to the k+1 nearest divided by k. d[0]=%f"%(d[0]))
        return sum(d[:self.k+1])/self.k # as the indiv is in the population, the first value is necessarily a 0.

    def size(self):
        return len(self.all_bd)

def updateNovelty(population, offspring, archive, encoder, params, population_saved=None):
   """Update the novelty criterion (including archive update)

   Implementation of novelty search following (Gomes, J., Mariano, P., & Christensen, A. L. (2015, July). Devising effective novelty search algorithms: A comprehensive empirical study. In Proceedings of GECCO 2015 (pp. 943-950). ACM.).
   :param population: is the set of indiv for which novelty needs to be computed
   :param offspring: is the set of new individuals that need to be taken into account to update the archive (may be the same as population, but it may also be different as population may contain the set of parents)
   :param params: dictionary containing run parameters. The relevant parameters are:
      * params["k"] is the number of nearest neighbors taken into account
      * params["add_strategy"] is either "random" (a random set of indiv is added to the archive) or "novel" (only the most novel individuals are added to the archive).
      * params ["lambda_nov"] is the number of individuals added to the archive for each generation
    :returns: The function returns the new archive
   """
   k=params["k"]
   add_strategy=params["add_strategy"]
   _lambda=params["lambda_nov"]

   for ind in population: ## Compute BD for each individual
       if (ind.bd is None):
           ind.bd = encoder(torch.as_tensor([ind.traj], dtype=torch.float).to(device), torch.as_tensor([ind.mask], dtype=torch.bool).to(device)).detach().cpu().numpy()[0]

   ref_pop = population

   # Novelty scores updates
   if (archive) and (archive.size()+len(ref_pop)>=k):
       if (verbosity(params,["all", "novelty"])):
           print("Update Novelty. Archive size=%d"%(archive.size()))

       for ind in population:
           if (True in np.isnan(ind.bd)):
               ind.novelty=-1
           else:
               ind.novelty=archive.get_nov(ind.bd, ref_pop)
   else:
       if (verbosity(params,["all", "novelty"])):
           print("Update Novelty. Initial step...")
       for ind in population:
           ind.novelty=0.

   if (verbosity(params,["all", "novelty"])):
       print("Fitness (novelty): ",end="")
       for ind in population:
           print("%.2f, "%(ind.novelty),end="")
       print("")
   if (len(offspring)<_lambda):
       print("ERROR: updateNovelty, lambda(%d)<offspring size (%d)"%(_lambda, len(offspring)))
       return None

   lbd=[]
   ltraj=[]
   lmask=[]
   # Update of the archive
   # we remove indivs with NAN values
   offspring2=list(filter(lambda x: not (True in np.isnan(x.traj)), offspring))
   if (len(offspring)!=len(offspring2)):
       print("WARNING: in updateNovelty, some individuals have a behavior descriptor with NaN values ! Initial offspring size: %d, filtered offspring size: %d"%(len(offspring), len(offspring2)))
   if (len(offspring2)<_lambda):
       print("WARNING: too few individuals have a non NaN value. We limit the number of added individuals to %d (number of offspring with non NaN bd)..."%(len(offspring2)))
       _lambda=len(offspring2)
   if(add_strategy=="random"):
       l=list(range(len(offspring2)))
       random.shuffle(l)
       if (verbosity(params,["all", "novelty"])):
           print("Random archive update. Adding offspring: "+str(l[:_lambda]))
       # lbd=[offspring2[l[i]].bd for i in range(_lambda)]
       ltraj=[offspring2[l[i]].traj for i in range(_lambda)]
       lmask=[offspring2[l[i]].mask for i in range(_lambda)]
   elif(add_strategy=="novel"):
       soff=sorted(offspring2,key=attrgetter("novelty"))
       ilast=len(offspring2)-_lambda
       # lbd=[soff[i].bd for i in range(ilast,len(soff))]
       ltraj=[soff[i].traj for i in range(ilast,len(soff))]
       lmask=[soff[i].mask for i in range(ilast,len(soff))]
       if (verbosity(params,["all", "novelty"])):
           print("Novel archive update. Adding offspring: ")
           for offs in soff[iLast:len(soff)]:
               print("    nov="+str(offs.novelty)+" fit="+str(offs.fitness.values)+" bd="+str(offs.traj))
   elif(add_strategy=="none"):
       # nothing to do...
       pass
   else:
       print("ERROR: updateNovelty: unknown add strategy(%s), valid alternatives are \"random\" and \"novel\""%(add_strategy))
       return None

   if(archive==None):
       # archive=NovArchive(lbd,k)
       archive=NovArchive(ltraj, lmask, encoder, k)
   else:
       archive.update(ltraj, lmask, encoder)

   return archive

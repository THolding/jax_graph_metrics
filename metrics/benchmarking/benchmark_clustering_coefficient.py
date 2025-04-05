#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 22:07:32 2025

"""

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt

import sys
from os import path
projectRootPath = path.join(path.abspath(path.dirname(__file__)), "..")
if projectRootPath not in sys.path:
    sys.path.append(projectRootPath)

import metrics



devices = jax.devices()
# if devices[0].device_kind != "GPU":
#     raise RuntimeError("No GPU detected")

ns = [10, 25, 50, 100, 250, 500, 750, 1000, 2000, 5000]
p = 0.5
numReps = 3

times = np.full((3, len(ns)), np.nan, dtype=float)
timesSd = np.full((3, len(ns)), np.nan, dtype=float)

for iN, N in enumerate(ns):
    print("N = ", N)
    nxGraph = nx.gnp_random_graph(N, p, seed=0)
    adj = nx.to_numpy_array(nxGraph)
    
    deviceAdj = jnp.array(adj)
    
    #networkx
    if N < 1000: #larger takes too long
        print("\tnetworkX")
        repTimes = []
        for rep in range(numReps):
             startTime = time.time()
             val = nx.average_clustering(nxGraph)
             repTimes.append(time.time() - startTime)
        times[0,iN] = np.mean(repTimes)
        timesSd[0,iN] = np.std(repTimes)
    
    #numpy
    print("\tnumpy")
    repTimes = []
    for rep in range(numReps):
         startTime = time.time()
         val = metrics.average_clustering_coefficient(adj, metrics.clustering_coefficient)
         repTimes.append(time.time() - startTime)
    times[1,iN] = np.mean(repTimes)
    timesSd[1,iN] = np.std(repTimes)
    
    #jax
    print("\tJAX")
    repTimes = []
    for rep in range(numReps):
         startTime = time.time()
         val = metrics.average_clustering_coefficient(deviceAdj, metrics.jax_clustering_coefficient)
         repTimes.append(time.time() - startTime)
    times[2,iN] = np.mean(repTimes)
    timesSd[2,iN] = np.std(repTimes)


labels = ["networkx", "numpy", "JAX"]
plt.figure()
for i in range(len(labels)):
    #plt.fill_between(ns, times[i,:]-timesSd[i,:], times[i,:]+timesSd[i,:], edgecolor='none', alpha=0.25)
    plt.plot(ns, np.log(times[i,:]), label=labels[i], linewidth=2)
plt.ylabel("time (log seconds)")
plt.xlabel("number of nodes")
plt.legend(loc=0)

#np.savetxt("cluster_coefficient_times.csv", times, delimiter=",")
#np.savetxt("cluster_coefficient_times_sd.csv", timesSd, delimiter=",")
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

runBenchmark = True
plotBenchmark = True

resultsDir = path.join("results")
ns = [10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500] #network sizes to use
p = 0.5 #probability of edge
numReps = 3 #number of repeats for each network size

if runBenchmark:
    devices = jax.devices()
    # if devices[0].device_kind != "GPU":
    #     raise RuntimeError("No GPU detected")
    
    times = np.full((3, len(ns)), np.nan, dtype=float)
    timesSd = np.full((3, len(ns)), np.nan, dtype=float)
    
    for iN, N in enumerate(ns):
        print("N = ", N)
        nxGraph = nx.gnp_random_graph(N, p)
        adj = nx.to_numpy_array(nxGraph)
        
        deviceAdj = jnp.array(adj)
        
        #networkx
        if N < 2000:#1000: #larger takes too long
            print("\tnetworkX")
            repTimes = []
            for rep in range(numReps):
                startTime = time.time()
                val = nx.average_clustering(nxGraph)
                repTimes.append(time.time() - startTime)
            times[0,iN] = np.mean(repTimes)
            timesSd[0,iN] = np.std(repTimes)
        
        #numpy
        if N < 2500:#1500: #larger takes too long
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
            print("\t\t", rep)
            startTime = time.time()
            val = metrics.average_clustering_coefficient(deviceAdj, metrics.jax_clustering_coefficient)
            repTimes.append(time.time() - startTime)
        times[2,iN] = np.mean(repTimes)
        timesSd[2,iN] = np.std(repTimes)
    
    
    np.savetxt(path.join(resultsDir, "clustering_coefficient_network_sizes.csv"), ns, delimiter=",")
    np.savetxt(path.join(resultsDir, "clustering_coefficient_times.csv"), times, delimiter=",")
    np.savetxt(path.join(resultsDir, "clustering_coefficient_times_sd.csv"), timesSd, delimiter=",")


if plotBenchmark:
    ns = np.genfromtxt(path.join(resultsDir, "clustering_coefficient_network_sizes.csv"), delimiter=",")
    times = np.genfromtxt(path.join(resultsDir, "clustering_coefficient_times.csv"), delimiter=",")
    labels = ["networkx", "numpy", "JAX (GPU)"]
    plt.figure()
    for i in range(len(labels)):
        #plt.plot(ns, np.log(times[i,:]), label=labels[i], linewidth=2)
        plt.plot(ns, times[i,:], label=labels[i], linewidth=2)
    plt.ylabel("time (log seconds)")
    plt.xlabel("number of nodes")
    plt.legend(loc=0)
    plt.savefig(path.join(resultsDir, "clustering_coefficient_times.pdf"))
    plt.savefig(path.join(resultsDir, "clustering_coefficient_times.png"), dpi=400)



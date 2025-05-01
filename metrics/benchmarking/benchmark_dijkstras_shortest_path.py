#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 07:09:47 2025

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
    
    times = np.full((2, len(ns)), np.nan, dtype=float)
    timesSd = np.full((2, len(ns)), np.nan, dtype=float)
    
    for iN, N in enumerate(ns):
        print("N = ", N)
        
        nxGraph = nx.gnp_random_graph(N, p, directed=True, seed=2)
        for (sourceId, destId) in nxGraph.edges():
            nxGraph[sourceId][destId]["weight"] = np.random.uniform(0.1, 10.0)
        nodeIds = np.arange(len(nxGraph))
        
        weights = nx.to_numpy_array(nxGraph, weight='weight')
        weights[weights==0] = np.inf

        deviceWeights = jnp.array(weights)
        
        #networkx
        if N <= 1000: #Larger networks take too long
            print("\tnetworkx")
            repTimes = []
            for irep in range(numReps):
                startTime = time.time()
                for sourceNode in nodeIds:
                    nxClosedOrder, nxPaths, nxPathCounts, nxShortestPathDistances = nx.algorithms.centrality.betweenness._single_source_dijkstra_path_basic(nxGraph, sourceNode, "weight")
                repTimes.append(time.time()-startTime)
            times[0,iN] = np.mean(repTimes)
            timesSd[0,iN] = np.std(repTimes)
        
        
        #jax
        print("\tjax")
        repTimes = []
        for irep in range(numReps):
            startTime = time.time()
            for sourceNode in nodeIds:
                jaxClosedOrder, jaxPaths, jaxShortestPathLengths, jaxPathCounts, jaxShortestPathDistances = metrics.jax_dijkstra_shortest_paths(deviceWeights, jnp.array(sourceNode))
            repTimes.append(time.time()-startTime)
        times[1,iN] = np.mean(repTimes)
        timesSd[1,iN] = np.std(repTimes)
    
    np.savetxt(path.join(resultsDir, "dijkstra_all_shortest_paths_network_sizes.csv"), ns, delimiter=",")
    np.savetxt(path.join(resultsDir, "dijkstra_all_shortest_paths_times.csv"), times, delimiter=",")
    np.savetxt(path.join(resultsDir, "dijkstra_all_shortest_paths_times_sd.csv"), timesSd, delimiter=",")



if plotBenchmark:
    ns = np.genfromtxt(path.join(resultsDir, "dijkstra_all_shortest_paths_network_sizes.csv"), delimiter=",")
    times = np.genfromtxt(path.join(resultsDir, "dijkstra_all_shortest_paths_times.csv"), delimiter=",")
    labels = ["networkx", "JAX (GPU)"]
    plt.figure()
    for i in range(len(labels)):
        #plt.plot(ns, np.log(times[i,:]), label=labels[i], linewidth=2)
        plt.plot(ns, times[i,:], label=labels[i], linewidth=2)
    plt.ylabel("time (seconds)")
    plt.xlabel("number of nodes")
    plt.title("clustering coefficient")
    plt.legend(loc=0)
    plt.savefig(path.join(resultsDir, "dijkstra_all_shortest_paths_times.pdf"))
    plt.savefig(path.join(resultsDir, "dijkstra_all_shortest_paths_times.png"), dpi=400)
        
    
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 07:08:33 2025

"""

import sys
from os import path
projectRootPath = path.join(path.abspath(path.dirname(__file__)), "..")
if projectRootPath not in sys.path:
    sys.path.append(projectRootPath)

import pytest
import networkx as nx
import numpy as np
import jax
import jax.numpy as jnp
import networkx.algorithms.centrality.betweenness as nxbetweenness

import metrics



def test_clustering_coefficient():
    np.random.seed(0)
    V = 10
    p = 0.5
    
    nxGraph = nx.gnp_random_graph(V, p, directed=True, seed=0)
    for (sourceId, destId) in nxGraph.edges():
        nxGraph[sourceId][destId]["weight"] = np.random.uniform(0.1, 10.0)
    nodeIds = np.arange(len(nxGraph))


    weights = nx.to_numpy_array(nxGraph, weight='weight')
    weights[weights==0] = np.inf
    deviceWeights = jnp.array(weights)
    sourceNode = nodeIds[0]
    

    ##networkx's CPU implementaton
    nxClosedOrder, nxPaths, nxPathCounts, nxShortestPathDistances = nx.algorithms.centrality.betweenness._single_source_dijkstra_path_basic(nxGraph, sourceNode, "weight")
    nxShortestPathDistances = [nxShortestPathDistances[i] for i in nodeIds]
    
    ##jax implementation
    jaxClosedOrder, jaxPaths, jaxShortestPathLengths, jaxPathCounts, jaxShortestPathDistances = metrics.jax_dijkstra_shortest_paths(deviceWeights, jnp.array(sourceNode))
    #convert shortest paths from a 2D buffered into a list of lists
    jaxPaths = np.array(jaxPaths)
    jaxPaths = [[jaxPaths[nodeId, i] for i in range(pathLength)] for nodeId, pathLength in enumerate(jaxShortestPathLengths)]
    
    #Compare the output of both version of the algorithm
    for i in nodeIds:
        ##Test closed orders match
        #Sometimes there are ties for the next node to process. This isn't an error, but does lead to small discrepencies between implementations.
        #Check for these and don't count them as failed tests here.
        #This only checks for ties of two, which are rare but do occur. It's possible but very unlikely that ties of three or more will occur. These are not errors but will cause the test to fail.
        closedOrderMatch = nxClosedOrder[i] == jaxClosedOrder[i]
        if closedOrderMatch == False: 
            if (nxClosedOrder[i-1] != jaxClosedOrder[i] and nxClosedOrder[i] != jaxClosedOrder[i+1]) or (nxClosedOrder[i+1] != jaxClosedOrder[i] and nxClosedOrder[i-1] != jaxClosedOrder[i]):
                closedOrderMatch == True
        assert closedOrderMatch
        
        ##Test shortest paths match
        assert nxPaths[i] == jaxPaths[i]
        
        ##Test shortest path counts match
        assert nxPathCounts[i] == jaxPathCounts[i]
        
        ##Test shortest path distances match
        assert nxShortestPathDistances[i] == pytest.approx(jaxShortestPathDistances[i], rel=1e-6)
        



        

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True) #64 bit precision
    
    pytest.main()

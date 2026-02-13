#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 07:09:31 2025

"""

import sys
from os import path
projectRootPath = path.join(path.abspath(path.dirname(__file__)), "..")
if projectRootPath not in sys.path:
    sys.path.append(projectRootPath)

import pytest
import networkx as nx
import jax;
import jax.numpy as jnp
import numpy as np

import metrics



def test_average_clustering_coefficient():
    V = 50 #number of nodes/vertices
    p = 0.25 #edge probability
    nxGraph = nx.gnp_random_graph(V, p, seed=0)
    weights = nx.to_numpy_array(nxGraph, weight='weight')
    weights[weights==0] = jnp.inf
    deviceWeights = jnp.array(weights)
    
    nodeIds = np.arange(len(nxGraph))
    
    sources = [0]
    destinations = [nodeId for nodeId in nodeIds]
    destinationsJax = jnp.full(nodeIds.shape, True)
    
    
    nxBCS = nx.betweenness_centrality_subset(nxGraph, sources, destinations, weight="weight", normalized=False)
    nxBCS = [nxBCS[i] for i in nodeIds]
    
    jaxBCS = metrics.jax_betweenness_centrality_subset(deviceWeights, sources, destinationsJax, directed=False)

    assert len(nxBCS) == len(jaxBCS)

    for i in range(len(nxBCS)):
        assert nxBCS[i] == pytest.approx(jaxBCS[i], rel=1e-6)



if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True) #64 bit precision
    
    #test_average_clustering_coefficient()
    pytest.main()



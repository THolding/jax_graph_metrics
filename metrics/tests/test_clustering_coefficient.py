#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 22:42:49 2025

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

import metrics


def test_clustering_coefficient():
    V = 50 #number of nodes/vertices
    p = 0.25 #edge probability
    nxGraph = nx.gnp_random_graph(V, p, seed=0)
    adj = nx.to_numpy_array(nxGraph)
    deviceAdj = jnp.array(adj)

    nxCcs = nx.clustering(nxGraph); nxCcs = [nxCcs[node] for node in sorted(nxCcs.keys())]


    for focalId in range(V):
        cc = metrics.jax_clustering_coefficient(deviceAdj, focalId)
        assert nxCcs[focalId] == pytest.approx(cc, rel=1e-6)


def test_average_clustering_coefficient():
    V = 50 #number of nodes/vertices
    p = 0.25 #edge probability
    nxGraph = nx.gnp_random_graph(V, p, seed=0)
    adj = nx.to_numpy_array(nxGraph)
    deviceAdj = jnp.array(adj)

    nxMeanCc = nx.average_clustering(nxGraph)
    
    meanCc = metrics.np_average_clustering_coefficient(adj)
    assert nxMeanCc == pytest.approx(meanCc, rel=1e-6)
    
    meanCc = metrics.jax_average_clustering_coefficient(deviceAdj)
    assert nxMeanCc == pytest.approx(meanCc, rel=1e-6)





if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True) #64 bit precision
    
    pytest.main()

    
    



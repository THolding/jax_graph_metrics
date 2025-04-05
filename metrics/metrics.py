#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 10:35:49 2025

"""

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import time

import jax
import jax.numpy as jnp


#Plain numpy implementation
def clustering_coefficient(adj, focalId):
    focalNeighbours = np.where(adj[focalId,:]!=0)[0]
    
    Kv = len(focalNeighbours) #number of neighborus
    
    sharedNeighbourCounts = []
    for neighbourId in focalNeighbours:
        sharedNeighbourCounts.append(np.sum(np.logical_and(adj[focalId,:], adj[neighbourId,:])))
    Nv = np.sum(sharedNeighbourCounts)
    
    cc = Nv / (Kv*(Kv-1)) #(2*Nv) / (Kv*(Kv-1))
    return cc



@jax.jit
def jax_clustering_coefficient(adj, focalId):
    focalNeighbourMask = adj[focalId,:]
    focalNeighbourCount = jnp.sum(focalNeighbourMask)
    
    sharedNeighbours = jnp.logical_and(adj, focalNeighbourMask)
    sharedNeighbourCounts = jnp.where(focalNeighbourMask,
                                      jnp.sum(sharedNeighbours, axis=1),
                                      0)
    cc = jnp.sum(sharedNeighbourCounts) / (focalNeighbourCount*(focalNeighbourCount-1))
    
    return cc



def average_clustering_coefficient(adj, clustering_func):
    N = adj.shape[0]
    meanCc = 0.0
    for focalId in range(N):
        meanCc += clustering_func(adj, focalId)
    return meanCc / N
    



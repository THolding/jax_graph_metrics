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
    







@jax.jit
def _jax_dijkstras_shortest_path_step(state):
    (isFrontier, isOpen, shortestPathDists, nodeIds, parents, shortestPathCounts, closedOrder, weights, shortestPathCounts, shortestPaths, shortestPathLengths, iteration) = state
    
    
    candidateCurrentNodeDistances = jnp.where(jnp.logical_and(isFrontier, isOpen), shortestPathDists, jnp.inf)
    currentNode = jnp.sum(jnp.argmin(candidateCurrentNodeDistances) > nodeIds)
    parentId = parents[currentNode]
    
    shortestPathCounts = shortestPathCounts.at[currentNode].set(shortestPathCounts[currentNode] + shortestPathCounts[parentId])  # count paths
    closedOrder = closedOrder.at[iteration].set(currentNode)
    
    
    #Calculate new distances to each neighbour
    isCurrentNodeNeighbours = weights[currentNode, :] != jnp.inf
    distSourceToCurrentNeighbours = weights[currentNode, :] + shortestPathDists[currentNode]
    
    
    ##Updating for new equal length shortest paths. Note: this must happen before updating for any new shorter shortest paths because otherwise shortestPathDists is updated to distSourceToCurrentNeighbours already...
    newPathIsEqual = jnp.logical_and(distSourceToCurrentNeighbours == shortestPathDists, shortestPathDists != jnp.inf)
    shortestPathCounts = jnp.where(newPathIsEqual, shortestPathCounts+shortestPathCounts[currentNode], shortestPathCounts)
    
    rows = jnp.where(newPathIsEqual, nodeIds, shortestPaths.shape[0]) #Note: shortestPaths.shape[0] is always out of bounds and will be clipped
    cols = jnp.where(newPathIsEqual, shortestPathLengths, shortestPaths.shape[1]) #Note: shortestPaths.shape[1] is always out of bounds and will be clipped
    shortestPaths = shortestPaths.at[rows, cols].set(currentNode)
    shortestPathLengths = jnp.where(newPathIsEqual, shortestPathLengths+1, shortestPathLengths)

    ##Updating for new shortest paths
    #Update currently known shortest path distances wherever we find new shortest distances
    newPathIsShortest = distSourceToCurrentNeighbours < shortestPathDists
    shortestPathDists = jnp.where(newPathIsShortest, distSourceToCurrentNeighbours, shortestPathDists)
    parents = jnp.where(newPathIsShortest, currentNode, parents) #Update parents of any node's shortest distance that was just updated
    shortestPathCounts = jnp.where(newPathIsShortest, 0, shortestPathCounts)
    
    rows = jnp.where(newPathIsShortest, nodeIds, shortestPaths.shape[0]) #Note: shortestPaths.shape[0] is always out of bounds and will be clipped
    shortestPaths = shortestPaths.at[rows, 0].set(currentNode)
    shortestPathLengths = jnp.where(newPathIsShortest, 1, shortestPathLengths)
    
    
    ##Update frontier
    isFrontier = isFrontier.at[currentNode].set(False) #Remove current node from the frontier
    isFrontier = jnp.logical_and(jnp.logical_or(isFrontier, isCurrentNodeNeighbours), isOpen) #Add any nodes which are newly accessible, but not closed, to the frontier
    
    
    isOpen = isOpen.at[currentNode].set(False)
    iteration = iteration+1
    
    
    
    return isFrontier, isOpen, shortestPathDists, nodeIds, parents, shortestPathCounts, closedOrder, weights, shortestPathCounts, shortestPaths, shortestPathLengths, iteration



@jax.jit
def _jax_dijkstras_shortest_paths_cond_func(val):
    return jnp.any(val[0]) #val[0] is the isFrontier array. Search must stop when there are no more nodes in the frontier. Note: this might be before all nodes are searched if it is not a fully connected graph



@jax.jit
def jax_dijkstra_shortest_paths(weights, sourceNode):
    NULL_NODE = -1
    nodeIds = jnp.arange(weights.shape[0])
    
    # modified from Eppstein
    closedOrder = jnp.full(nodeIds.shape, NULL_NODE) #settles order (order in which the nodes were closed)
    shortestPaths = jnp.full((nodeIds.shape[0], nodeIds.shape[0]), NULL_NODE)
    shortestPathLengths = jnp.full(nodeIds.shape, 0)
    
    shortestPathDists = jnp.full(nodeIds.shape, np.inf)
    shortestPathDists = shortestPathDists.at[sourceNode].set(0.0)
    shortestPathCounts = jnp.full(nodeIds.shape, 0)
    shortestPathCounts = shortestPathCounts.at[sourceNode].set(1)
    
    parents = jnp.full(nodeIds.shape, NULL_NODE)
    parents = parents.at[sourceNode].set(sourceNode)
    
    isFrontier = jnp.full(nodeIds.shape, False)
    isFrontier = isFrontier.at[sourceNode].set(True)
    
    isOpen = jnp.full(nodeIds.shape, True)
    
    iteration = jnp.array(0)
    
    
    #Iteratively solve until the frontier is empty (maximum V iterations)
    (isFrontier, isOpen, shortestPathDists, nodeIds, parents, shortestPathCounts, closedOrder,
     weights, shortestPathCounts, shortestPaths, shortestPathLengths, iteration) = jax.lax.while_loop(_jax_dijkstras_shortest_paths_cond_func,
                                                                                                      _jax_dijkstras_shortest_path_step,
                                                                                                      (isFrontier, isOpen, shortestPathDists, nodeIds, parents, shortestPathCounts, closedOrder,
                                                                                                       weights, shortestPathCounts, shortestPaths, shortestPathLengths, iteration)
                                                                                                      )
    
    return closedOrder, shortestPaths, shortestPathLengths, shortestPathCounts, shortestPathDists #closedOrder is the order in which each node was closed (used in Brandes' algorithm), P is the paths taken, shortestPathCounts is the shortest path counts, D is distances




###########Untested: nx_accumulate_subset and nx_betweenness_centrality_subset
def nx_accumulate_subset(betweenness, backPropgationOrder, shortestPaths, shortestPathCounts, sourceNode, destinations):
    delta = dict.fromkeys(backPropgationOrder, 0.0)
    destinations = set(destinations) - {sourceNode}
    while len(backPropgationOrder) > 0:
        currentNode = backPropgationOrder.pop()
        if currentNode in destinations:
            coeff = (delta[currentNode] + 1.0) / shortestPathCounts[currentNode]
        else:
            coeff = delta[currentNode] / shortestPathCounts[currentNode]
        for nodeInPath in shortestPaths[currentNode]:
            delta[nodeInPath] += shortestPathCounts[nodeInPath] * coeff
        if currentNode != sourceNode:
            betweenness[currentNode] += delta[currentNode]
    return betweenness


sources = [0]
destinations = [2]

def nx_betweenness_centrality_subset(weights, sources, destinations, directed):
    betweenness = np.full(weights.shape[0], 0.0)
    for source in sources:
        closeOrder, shortestPaths, shortestPathCounts, _ = nx_single_source_dijkstra_path_basic(weights, source)
        betweenness = nx_accumulate_subset(betweenness, closeOrder, shortestPaths, shortestPathCounts, source, destinations)
    
    if directed:
        betweenness /= 2
    return betweenness



np.random.seed(0)
N = 5#10#50#5
p = 0.75#0.65#0.45#0.75
nxGraph = nx.gnp_random_graph(N, p, directed=True, seed=2)
for (sourceId, destId) in nxGraph.edges():
    nxGraph[sourceId][destId]["weight"] = np.round(np.random.uniform(0.1, 6.0), 5)
nodeIds = np.arange(len(nxGraph))

#overwrite some weights to make two shortest distance paths
nxGraph[0][4]["weight"] = 1.5
nxGraph[4][1]["weight"] = 2.5
nxGraph[0][3]["weight"] = 1.5
nxGraph[3][1]["weight"] = 2.5

edgeLabels = {}
for (sourceId, destId) in nxGraph.edges():
    label = str((sourceId, destId))+"="+str(nxGraph[sourceId][destId]["weight"])
    if nxGraph.has_edge(destId, sourceId):
        label += "\n"+str((destId, sourceId))+"="+str(nxGraph[destId][sourceId]["weight"])
    edgeLabels[(sourceId, destId)] = label
    

pos = nx.spring_layout(nxGraph)
plt.figure()
nx.draw(nxGraph, pos, with_labels=True, node_size=500, arrows=True)
nx.draw_networkx_edge_labels(nxGraph, pos, edge_labels=edgeLabels)

weights = nx.to_numpy_array(nxGraph, weight='weight')
weights[weights==0] = np.inf

deviceWeights = jnp.array(weights)


sourceNode = nodeIds[0]



jax_dijkstra_shortest_paths(deviceWeights, sourceNode)


###################



# nxBetweennewssCentrality = nx.betweenness_centrality_subset(nxGraph, [0], [1], weight="weight")
# nxBetweennewssCentrality = [nxBetweennewssCentrality[i] for i in nodeIds]












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
def np_clustering_coefficient(adj, focalId):
    focalNeighbours = np.where(adj[focalId,:]!=0)[0]
    
    Kv = len(focalNeighbours) #number of neighborus
    
    sharedNeighbourCounts = []
    for neighbourId in focalNeighbours:
        sharedNeighbourCounts.append(np.sum(np.logical_and(adj[focalId,:], adj[neighbourId,:])))
    Nv = np.sum(sharedNeighbourCounts)
    
    cc = Nv / (Kv*(Kv-1)) #(2*Nv) / (Kv*(Kv-1))
    return cc



#Returns the clustering coefficient for a single node on a network
#Supports directed and undirected graphs with no weighted edges
#adj: the adjacency matrix for the network, where 0 indicates no edge, 1 indicates an edge between the row-indexed to the column-indexed node
#focalId: index of the node to calculate the clustering coefficient for
@jax.jit
def jax_clustering_coefficient(adj, focalId):
    focalNeighbourMask = adj[focalId,:] #identify neighbours
    focalNeighbourCount = jnp.sum(focalNeighbourMask) #number of neighbours
    
    sharedNeighbours = jnp.logical_and(adj, focalNeighbourMask) #Boolean array indicating which neighbour nodes share another neighbour with the focal node
    sharedNeighbourCounts = jnp.where(focalNeighbourMask,
                                      jnp.sum(sharedNeighbours, axis=1),
                                      0)
    #Sum the number of shared other-neighbours for each neighbour
    cc = jnp.sum(sharedNeighbourCounts) / (focalNeighbourCount*(focalNeighbourCount-1))
    
    return cc



def np_average_clustering_coefficient(adj):
    N = adj.shape[0]
    nodeIds = np.arange(N)
    #meanCc = 0.0
    nodeCCs = np.vectorize(np_clustering_coefficient, excluded=[0])(adj, nodeIds)
    # for focalId in range(N):
    #     meanCc += clustering_coefficient(adj, focalId)
    return np.sum(nodeCCs) / N
    

@jax.jit
def jax_average_clustering_coefficient(adj):
    nodeIds = jnp.arange(adj.shape[0])
    nodeCCs = jax.vmap(jax_clustering_coefficient, (None, 0))(adj, nodeIds)
    return jnp.sum(nodeCCs) / nodeIds.shape[0]







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





@jax.jit
def _jax_accumulate_subset_cond_func(state):
    backPropagationIdx = state[0]
    return backPropagationIdx[0] >= 0


@jax.jit
def _jax_accumulate_subset_step(state):
    (backPropagationIdx, betweenness, backPropagationOrder, shortestPaths, shortestPathCounts, sourceNode, destinations, dependencyScores, NULL_NODE) = state
    
    #If we're in the buffered part of the array, just move on to the next element
    currentNode = backPropagationOrder[backPropagationIdx]
    
    dummyNotNull = int(currentNode.at[0] != NULL_NODE.at[0])
    
    #Calculate coefficient
    coeff = dependencyScores[currentNode]
    coeff = coeff + jnp.array(destinations[currentNode], dtype=int) #Add 1 to numerator if current node is a destination
    coeff = coeff / shortestPathCounts[currentNode]
    coeff = coeff * dummyNotNull #multiplying by dummyNotNull results in the function making no changes (effectively skipping to the next loop iteration). #TODO: use jax.jax.cond instead?
    # jax.debug.print("coeff4: {0} {1}", currentNode, coeff)
    
    #Update dependency scores
    nodesInPath = shortestPaths[currentNode].squeeze()
    unorderedPathCounts = jnp.where(nodesInPath != NULL_NODE, shortestPathCounts[nodesInPath], 0)
    pathCounts = jnp.zeros(unorderedPathCounts.shape)
    pathCounts = pathCounts.at[nodesInPath].set(unorderedPathCounts)
    # jax.debug.print("pathCounts: {0} {1}", currentNode, pathCounts)
    newDependencyContributions = pathCounts * coeff
    # jax.debug.print("newDependencyContributions: {0} {1}", currentNode, newDependencyContributions)
    # jax.debug.print("dependencyScores before: {0} {1}", currentNode, dependencyScores)
    dependencyScores = dependencyScores + newDependencyContributions
    # jax.debug.print("dependencyScores after: {0} {1}", currentNode, dependencyScores)

    
    #Avoid if conditional by using a dummy scalar
    dummy = currentNode != sourceNode
    betweenness = betweenness.at[currentNode].set(betweenness[currentNode] + (dummy*dependencyScores[currentNode]))
    
    backPropagationIdx = backPropagationIdx - 1
    
    return backPropagationIdx, betweenness, backPropagationOrder, shortestPaths, shortestPathCounts, sourceNode, destinations, dependencyScores, NULL_NODE



#The back propagation step in Brandes' algorithm
#destinations: a jnp.array boolean mask with a length of V (number of vertices in whole graph), where True indicates a destination node.
@jax.jit
def jax_accumulate_subset(betweenness, backPropagationOrder, shortestPaths, shortestPathCounts, sourceNode, destinations):
    NULL_NODE = -1
    dependencyScores = jnp.full(backPropagationOrder.shape, 0.0)
    
    destinations = destinations.at[sourceNode].set(False) #source cannot be a destination
    backPropagationIdx = jnp.array([backPropagationOrder.shape[0]-1])
    
    
    state = jax.lax.while_loop(_jax_accumulate_subset_cond_func,
                               _jax_accumulate_subset_step,
                               (backPropagationIdx, betweenness, backPropagationOrder, shortestPaths, shortestPathCounts, sourceNode, destinations, dependencyScores, NULL_NODE)
                              )
    
    betweenness = state[1]
    
    return betweenness






#The internal function used in while loop in jax_betweeness_centrality_subset
@jax.jit
def _jax_betweenness_centrality_subset_step(state):
    (sourceIdx, weights, sources, destinations, betweenness) = state
    
    source = sources[sourceIdx]
    
    closeOrder, shortestPaths, shortestPathLengths, shortestPathCounts, _ = jax_dijkstra_shortest_paths(weights, source)
    betweenness = jax_accumulate_subset(betweenness, closeOrder, shortestPaths, shortestPathCounts, source, destinations)
    
    sourceIdx = jnp.array(sourceIdx - 1)
    return (sourceIdx, weights, sources, destinations, betweenness)

@jax.jit
def _jax_betweenness_centrality_subset_cond_func(state):
    sourceIdx = state[0]
    return sourceIdx >= 0
    
#An implementaton of Brandes' agorithm
#destinations: a jnp.array boolean mask with a length of V (number of vertices in whole graph), where True indicates a destination node.
#This is a non-jax function which calls jax functions to do the heavy lifting
def jax_betweenness_centrality_subset(weights, sources, destinations, directed):
    nodeIds = jnp.arange(weights.shape[0])
    sourceIdx = jnp.array(len(sources)-1) #Start indexing from the right
    
    #create a buffered sources array so that different sized arrays don't force recompilation
    bufferedSources = jnp.where(nodeIds <= sourceIdx,
                                jnp.array(sources),
                                -1)
    
    betweenness = jnp.full(weights.shape[0], 0.0)
    
    state = jax.lax.while_loop(_jax_betweenness_centrality_subset_cond_func,
                               _jax_betweenness_centrality_subset_step,
                               (sourceIdx, weights, bufferedSources, destinations, betweenness))
    betweenness = state[-1]

    
    if directed == False:
        betweenness /= 2
    return betweenness






### Temporary development code
if __name__ == "__main__":
    np.random.seed(0)
    N = 20#10#50#5  #E.G. set N=10 and betweenness centrality fails fails
    p = 0.25#0.65#0.45#0.75
    nxGraph = nx.gnp_random_graph(N, p, directed=True, seed=2)
    for (sourceId, destId) in nxGraph.edges():
        nxGraph[sourceId][destId]["weight"] = np.round(np.random.uniform(0.1, 6.0), 5)
    nodeIds = np.arange(len(nxGraph))
    
    #overwrite some weights to make two shortest distance paths
    # nxGraph[0][4]["weight"] = 1.5
    # nxGraph[4][1]["weight"] = 2.5
    # nxGraph[0][3]["weight"] = 1.5
    # nxGraph[3][1]["weight"] = 2.5
    
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
    
    
    source = 0
    destinations = [nodeId for nodeId in nodeIds] #[2]
    
    destinationsJax = jnp.full(nodeIds.shape, False)
    for dest in destinations:
        destinationsJax = destinationsJax.at[dest].set(True)
    
    
    
    
    #### Testing accumulate_subset
    S, P, sigma, _ = nx.algorithms.centrality.betweenness._single_source_dijkstra_path_basic(nxGraph, source, "weight")
    b = dict.fromkeys(nxGraph, 0.0) 
    nxBCS = nx.algorithms.centrality.betweenness_subset._accumulate_subset(b, S, P, sigma, source, destinations)
    nxBCS = [nxBCS[i] for i in nodeIds]
    
    closeOrder, shortestPaths, shortestPathLengths, shortestPathCounts, _ = jax_dijkstra_shortest_paths(deviceWeights, source)
    b = jnp.full(weights.shape[0], 0.0)
    jaxBCS = jax_accumulate_subset(b, closeOrder, shortestPaths, shortestPathCounts, jnp.array([source]), destinationsJax)
    
    print("ID, nx, jax")
    for i in range(len(nxBCS)):
        print(i, nxBCS[i], jaxBCS[i])
        if nxBCS[i] != jaxBCS[i]:
            print("################# MISMATCH")
    
    
    
    
    #### Testing betweenness_centrality_subset
    nxBCS = nx.betweenness_centrality_subset(nxGraph, [source], destinations, weight="weight", normalized=False)
    nxBCS = [nxBCS[i] for i in nodeIds]
    
    
    jaxBCS = jax_betweenness_centrality_subset(deviceWeights, [source], destinationsJax, directed=True)
    
    print("\n\nID, nx, jax")
    for i in range(len(nxBCS)):
        print(i, nxBCS[i], jaxBCS[i])
        if nxBCS[i] != jaxBCS[i]:
            print("################# MISMATCH")













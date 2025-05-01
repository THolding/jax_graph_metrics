#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 07:26:09 2025

"""

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


#weights (n by n matrix of edge weights)
#categories: categories that each node belongs to (array of length n)
def np_modularity(weights, categories):
    nodeIds = jnp.arange(weights.shape[0])
    
    modularity = 0.0
    for inode in range(len(nodeIds)):
        focalCat = categories[inode]
        
        neighbourSharesCat = jnp.logical_and(weights[inode,:] != jnp.inf, categories == focalCat)
        
        totalSharedCat = jnp.sum(neighbourSharesCat)
        
        #calculate random expected shared cat neighbours
        numNeighbours = jnp.sum(weights[inode,:] != jnp.inf)
        
        #mask for possible edge connections
        noneSelfEdges = jnp.full(nodeIds.shape, True)
        noneSelfEdges = noneSelfEdges.at[inode].set(False)
        #scale for possible edge connections (how many edges does each other node have?)
        numEdgesByNode = np.array([np.sum(weights[i,:] != jnp.inf) for i in range(len(nodeIds))])
        totalNumEdges = jnp.sum(numEdgesByNode)
        #ignore edges from self and non-self communities
        numEdgesByNode[inode] = 0
        numEdgesByNode[communities==focalCat] = 0
        
        #calculate the expected number of self-community edge connections, given the number of edges each node has
        expectedSharedCommunityNeighbours = 0
        for jnode in range(len(nodeIds)):
            expectedSharedCommunityNeighbours = (numNeighbours*numEdgesByNode[jnode])/((2*totalNumEdges) - 1)
        
        
        #calculate modularity
        modularity += totalSharedCat - expectedSharedCommunityNeighbours
    return modularity#/(2*totalNumEdges)
        



def _nx_community_contribution(communityNodes, weights, weightStr, outDegree, inDegree, resolution, m, norm):
    #sum of all intra-commuity edge weights
    from itertools import product
    idx = list(product(communityNodes, communityNodes))
    x = [v1 for v1, v2 in idx]
    y = [v2 for v1, v2 in idx]
    idx = (x, y)
    communityWeightsSum = sum(weights[idx])/2 #divide by 2 because otherwise we're counting the weight twice (once form u to v, once from v to u)
    
    outDegreeSum = sum(outDegree[communityNodes])
    inDegreeSum = sum(inDegree[communityNodes])

    return communityWeightsSum / m - resolution * outDegreeSum * inDegreeSum * norm

def nx_modularity(G, weightStr, weights, communities, resolution=1, directed=False):
    # if directed:
    #     out_degree = dict(G.out_degree(weight=weight))
    #     in_degree = dict(G.in_degree(weight=weight))
    #     m = sum(out_degree.values())
    #     norm = 1 / m**2
    # else:
    #undirected
    outDegree = np.sum(weights, axis=1)
    inDegree = outDegree
    weightsSum = sum(outDegree)
    m = weightsSum / 2
    norm = 1 / weightsSum**2

    modularity = 0.0
    for communityId in range(len(communities)):
        modularity += _nx_community_contribution(communities[communityId], weights, weightStr, outDegree, inDegree, resolution, m, norm)
    
    return modularity

    #return sum(map(_nx_community_contribution, communities, weights, weightStr, outDegree, inDegree, resolution, m, norm))
        
        



np.random.seed(0)
N = 5#20#10#50#5
p = 0.5#0.65#0.45#0.75

nxGraph = nx.gnp_random_graph(N, p, directed=False, seed=2)
# for (sourceId, destId) in nxGraph.edges():
#     nxGraph[sourceId][destId]["weight"] = np.round(np.random.uniform(0.1, 6.0), 5)
nodeIds = np.arange(len(nxGraph))

#overwrite some weights to make two shortest distance paths
# nxGraph[0][4]["weight"] = 1.5
# nxGraph[4][1]["weight"] = 2.5
# nxGraph[0][3]["weight"] = 1.5
# nxGraph[3][1]["weight"] = 2.5

# edgeLabels = {}
# for (sourceId, destId) in nxGraph.edges():
#     label = str((sourceId, destId))+"="+str(nxGraph[sourceId][destId]["weight"])
#     if nxGraph.has_edge(destId, sourceId):
#         label += "\n"+str((destId, sourceId))+"="+str(nxGraph[destId][sourceId]["weight"])
#     edgeLabels[(sourceId, destId)] = label

numCommunities = 2
communities = np.random.choice(np.arange(numCommunities), N, replace=True)
communityLists = [np.where(communities==i)[0].tolist() for i in range(numCommunities)]

pos = nx.spring_layout(nxGraph)
# plt.figure()
# nx.draw(nxGraph, pos, with_labels=True, node_size=500, arrows=True, node_color=communities)
# nx.draw_networkx_edge_labels(nxGraph, pos)#, edge_labels=edgeLabels)

weights = nx.to_numpy_array(nxGraph, weight='weight')
#weights[weights==0] = np.inf

deviceWeights = jnp.array(weights)



nxModularity = nx.algorithms.community.modularity(nxGraph, communities=communityLists, weight=None)
#npModularity = np_modularity(weights, communities)
npnx2Modularity = nx_modularity(nxGraph, "weight", weights, communityLists, directed=False)

print("modularity nx:    ", nxModularity)
print("modularity npnx2: ", npnx2Modularity)




communityMask = communities == 0
rowMask = communityMask[:, None]
colMask = communityMask[None, :]
communityWeightsMask = jnp.outer(communityMask, colMask)
communityWeightsMask.shape

TODO: use this in the _nx_community_contribution function - Test separately!










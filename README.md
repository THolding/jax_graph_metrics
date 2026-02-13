# jax_graph_metrics
This repository provides example parallelised implementations some graph metrics, namely clustering coefficient and betweenness centrality, using `jax` in Python. It is intended to demonstrate the potential performance gain over equivalent implementations using `numpy` or provided in the `networkx` package.

The end-user functions are stored in `metrics.py`. See inline comments and example code for how to use them.
 - metrics.jax_clustering_coefficient
 - metrics.jax_average_clustering_coefficient
 - metrics.jax_dijkstra_shortest_paths
 - metrics.jax_betweenness_centrality_subset


Pytest tests included in the `tests` directory. These compare (jax or numpy) implementation output to the output from the equivalent `networkx` functions. Benchmarking against networkx and numpy implementations are also provided.

Contains JAX JIT optimised functions for:
 - Clustering coefficient
   - working, tested, benchmarked
 - Dikstras's shortest paths
   - working, tested, benchmarked
 - Betweenness centrality
   - mostly working, but there's a bug for certain edge-cases
   - simple test code written
   - no benchmarking code
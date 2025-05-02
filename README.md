# jax_graph_metrics
Prototyping jax functions for calculating graph metrics.

Main file is `metrics.py`. The main functions intended for end user are:
 - metrics.jax_clustering_coefficient
 - metrics.jax_average_clustering_coefficient
 - metrics.jax_dijkstra_shortest_paths
 - metrics.jax_betweenness_centrality_subset


Pytest tests included in the `tests` directory which compare to expected output from the equivalent `networkx` functions. Benchmarking against networkx and 

Contains JAX JIT optimised functions for:
 - Clustering coefficient
  - working, tested, benchmarked
 - Dikstras's shortest paths
  - working, tested, benchmarked
 - Betweenness centrality
  - mostly working: still a bug
  - simple test code written
  - no benchmarking code

Documentation is lacking, but is hopefully self-explanatory from the testing code. Feel free to ask me though.

One thing which might need changing is that I use jnp.inf rather than 0 for the adjacency matrix when there is no edge between two nodes. But this is probably trivial to change as none of the algorithms really rely on that except when creating masks.
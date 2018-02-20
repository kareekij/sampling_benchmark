# Benchmarking different sampling algorithms
The goal is to study how network structural properties affect the algorithms performance. 
Three networks properties are considered, 1) Modularity 2) Average degree and 3) Average community size.

## Requirements
* python
* networkx
* numpy
* scipy
* Create folder `log` within this folder

## 9 popular algorithms are implemented
* Breadth-first Search Sampling
* Depth-first Search Sampling
* Snowball Sampling
* Random Sampling
* Maximum Observed Degree Sampling
* Maximum PageRank Sampling
* Online Page Importance Computation
* Volatile Multi-armed Bandit
* Random Walk Sampling


### How to run
`python sample.py <path_to_file> -b <budget> -experiment <number_of_experiments>`

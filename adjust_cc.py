from __future__ import division, print_function
import numpy as np
import networkx as nx
import csv
import sys
import random
import operator
import time
import math
import argparse

import oracle
import query
import log


import community
import _mylib
import Queue
from scipy import stats


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('cc', help='expected cc', type=float)

	args = parser.parse_args()
	print(args)
	expected_cc = args.cc


	#expected_cc = 0.15
	PROB_ADD = 0.8

	G = nx.read_edgelist('./data/mix-param/0.2/network.dat')
	partition = community.best_partition(G)
	print(nx.info(G))
	current_cc = nx.average_clustering(G)
	print('current cc', current_cc)

	while current_cc > expected_cc:
	#while current_cc < expected_cc:
		for p in set(partition.values()):
			members = _mylib.get_members_from_com(p, partition)
			s = G.subgraph(members)
			e_count = s.number_of_edges()
			k = int(.1 * (e_count))

			for i in range(0, k):
				e_list = random.sample(s.edges(),2)
				e1 = e_list[0]
				e2 = e_list[1]
				G.remove_edges_from(e_list)
				G.add_edge(e1[0],e2[0])
				G.add_edge(e1[1],e2[1])
				r = random.uniform(0,1)
				if r <= PROB_ADD:
					nodes = random.sample(G.nodes(),2)
					G.add_edge(nodes[0], nodes[0])
				s = G.subgraph(members)

			# for i in range(0,k):
			# 	e_list = random.sample(s.edges(),2)
			# 	e1 = e_list[0]
			# 	e2 = e_list[1]
			# 	G.remove_edges_from(e_list)
			# 	G.add_edge(e1[0],e2[0])
			# 	G.add_edge(e1[1],e2[1])
			# 	r = random.uniform(0,1)
			# 	if r <= PROB_ADD:
			# 		nodes = random.sample(G.nodes(),2)
			# 		G.add_edge(nodes[0], nodes[0])
			# 	s = G.subgraph(members)

		current_cc = nx.average_clustering(G)
		current_edge_count = G.number_of_edges()
		print('current cc: {}, edge: {}'.format(current_cc, current_edge_count))

	print('-'*10)
	partition = community.best_partition(G)

	print('# edges', G.number_of_edges())

	print(nx.info(G))
	nx.write_edgelist(G, "./network.dat")


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
import query_cheaper
import log


import community
import _mylib
import Queue
from scipy import stats


if __name__ == '__main__':
	G = nx.read_edgelist('./data/mix-param/0.4/network.dat')
	partition = community.best_partition(G)
	print(nx.info(G))
	print('cc', nx.average_clustering(G))

	for p in set(partition.values()):
		members = _mylib.get_members_from_com(p, partition)
		s = G.subgraph(members)
		e_count = s.number_of_edges()
		k = int(.1 * (e_count))
		#print('shuffle {} edges'.format(k))
		for i in range(0,k):
			e_list = random.sample(s.edges(),2)
			e1 = e_list[0]
			e2 = e_list[0]
			G.remove_edges_from(e_list)
			G.add_edge(e1[0],e2[0])
			G.add_edge(e1[1],e2[1])

	print('-'*10)
	partition = community.best_partition(G)
	print(nx.info(G))
	cc = nx.average_clustering(G)
	print('cc after', cc)
	# # #if cc < 0.3:
	#nx.write_edgelist(G, "./data/mix-param/0.4/network_" + str(cc) + '.dat')


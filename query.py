# -*- coding: utf-8 -*-

"""
Simulate the API queries
"""

import networkx as nx
import random
import _mylib
import pickle
import community
import os

class UndirectedSingleLayer(object):
	"""
	Class to simulate API queries for undirected single layer graphs
	"""
	def __init__(self, graph, cost=1):
		super(UndirectedSingleLayer, self).__init__()
		self._graph = graph 				# The complete graph
		self._cost_neighbor = cost 			# Cost of each query (default: 1)

	def neighbors(self, node):
		"""
		Return the neighbors of a node

		Args:
			node (str) -- The node id whose neighbors are needed
		Return:
			list[str] -- List of node ids which are neighbors of node
		"""
		nodes = self._graph.neighbors(node)
		edges = [(node, n) for n in nodes]

		return set(nodes), set(edges), self._cost_neighbor

	def randomNode(self):
		"""
		Return a random node from the graph

		Args:
			None
		Return:
			str -- The node id of a random node in the graph
		"""
		nodes = self._graph.nodes()
		return random.choice(nodes)

	def randomHighDegreeNode(self):
		degree = self._graph.degree()
		degree_sorted = _mylib.sortDictByValues(degree,reverse=True)
		size = int(.3 * len(degree))
		degree_sorted = degree_sorted[:size]
		return random.choice(degree_sorted)[0]

	def randomFarNode(self):
		degree = self._graph.degree()
		deg_one_nodes = _mylib.get_members_from_com(2,degree)
		cc = nx.clustering(self._graph)

		for n in deg_one_nodes:
			print(cc[n])

	def randomFromLargeCommunity(self, G, dataset):
		com_fname = './data/pickle/communities_{}.pickle'.format(dataset)
		if os.path.isfile(com_fname):
			partition = pickle.load(open(com_fname, 'rb'))
		else:
			partition = community.best_partition(G)
			pickle.dump(partition, open(com_fname, 'wb'))

		count_members = {}
		for p in set(partition.values()):
			members = _mylib.get_members_from_com(p, partition)
			count_members[p] = len(members)

		selected_p, i = _mylib.get_max_values_from_dict(count_members, count_members.keys())
		members = _mylib.get_members_from_com(selected_p, partition)

		degree = self._graph.degree(members)
		degree_sorted = _mylib.sortDictByValues(degree, reverse=True)
		size = int(.5 * len(degree))
		degree_sorted = degree_sorted[:size]
		return random.choice(degree_sorted)[0]





# -*- coding: utf-8 -*-

"""
Script for the sampling algorithm
"""

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

from sklearn import linear_model

starting_node = -1

class UndirectedSingleLayer(object):
	"""
	Class for Expansion-Densification sammling
	"""


	def __init__(self, query, budget=100, bfs_count=10, exp_type='oracle', dataset=None, logfile=None,k=5,cost=False, log_int=10):
		super(UndirectedSingleLayer, self).__init__()
		self._budget = budget 			# Total budget for sampling
		self._bfs_count = bfs_count 	# Portion of the budget to be used for initial bfs
		self._query = query 			# Query object
		self._dataset = dataset 		# Name of the dataset used; Used for logging and caching
		self._logfile = logfile 		# Name of the file to write log to
		self._exp_type = exp_type
		self._k = k
		self._isCost = cost
		self._log_interval = log_int
		self._stage = None
		self._one_cost = 0
		self._one_gain = 0

		self._cost = 0 					# Keep track of the cost spent
		self._sample = {'edges': set(), 'nodes':\
		 {'close':set(), 'open':set()}}
		self._wt_exp = 1 				# Weight of the expansion score
		self._wt_den = 3 				# Weight of the densification score
		self._score_den_list = [] 			# List to store densification scores; used only for logging
		self._score_exp_list = [] 			# List to store expansion scores; used only for logging
		self._new_nodes = []			# List of the new nodes observed at each densification
		self._cumulative_new_nodes = [] # Cummulated new nodes
		self._exp_cut_off = 50			# Numbor of expansion candidates
		self._den_cut_off = 100 			# Number of densificaiton candidates
		self._sample_graph = nx.Graph() # The sample graph
		self._nodes_observed_count = [] # Number of nodes observed in each iteration
		self._avg_deg = 0.
		self._med_deg = 0.
		self._cost_spent = []
		self._nodes_return = []
		self._exp_count = 0
		self._densi_count = 0
		self._track_obs_nodes = []
		self._track_cost = []
		self._track = {}
		self._track_cc = {}

		self._track_k = []
		self._track_open= []

		self._tmp = 0
		self._avg = {'unobs':[], 'close':[], 'open':[]}

		self._percentile = 90
		self._sd = []

		self._line_1 = []
		self._line_2 = []

		self._X = []
		self._Y = []



	def _expansion_random(self, candidate_list):
		return random.sample(candidate_list,1)[0]

	def _expansion_random_deg_mt_one(self, candidate_list):
		g = nx.Graph()
		g.add_edges_from(self._sample['edges'])
		g = _mylib.remove_node_with_deg(g)

		candidate_list = self._sample['nodes']['open'] - set(g.nodes())
		return random.sample(candidate_list, 1)[0]

	def _sample_open_node(self, candidate_list):
		count = 1
		DECAY_R = 0.005
		GROWTH_R = 0.01
		node_unobs_count = {}
		node_unobs = {}
		while self._cost < self._budget:

			degree = self._sample_graph.degree(self._sample['nodes']['close'])
			degree_seq = sorted(degree.values())

			sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
			open_nodes = list(self._sample['nodes']['open'])

			node = random.choice(open_nodes)
			sub_sample['nodes']['open'].add(node)
			# Make a query
			nodes, edges, c2 = self._query.neighbors(node)

			obs_nodes = set(self._sample['nodes']['open']).union(set(self._sample['nodes']['close']))
			unobs_nodes = nodes - obs_nodes

			# TODO:
			self._new_nodes.append(len(unobs_nodes))

			node_deg = len(nodes)
			percent_tile = stats.percentileofscore(degree_seq, node_deg)
			node_unobs_count[node] = len(unobs_nodes)
			node_unobs[node] = unobs_nodes
			#percent_tile = stats.percentileofscore(self._den_unobs, len(unobs_nodes))

			print('	  {}	Percentile {}/{} -- deg {}/{}'.format( len(open_nodes), percent_tile,self._percentile, len(unobs_nodes), node_deg ))

			self._increment_cost(c2)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, node)
			self._updateSample(sub_sample)

			if percent_tile > self._percentile:
				self._percentile = self._percentile * math.pow((1 + GROWTH_R), count)
				break
			else:
				self._percentile = self._percentile*math.pow((1-DECAY_R), count)
				#count += 1

		obs_nodes = set(self._sample['nodes']['open']).union(set(self._sample['nodes']['close']))

		node_unobs_sort = _mylib.sortDictByValues(node_unobs_count, reverse=True)
		best_node = node_unobs_sort[0][0]
		best_node_unobs = node_unobs_sort[0][1]
		# Already query, not neccessary to increment cost
		neighbors_unobs = node_unobs[best_node]
		# Pick one from the open nodes
		candidates = neighbors_unobs - set(self._sample['nodes']['close'])

		if len(candidates) == 0:
			print('  -- no candidates from this node, pick from any open nodes')
			candidates = self._sample['nodes']['open']

		print(' -- # best unobs {} - candidate {} / {}'.format(best_node_unobs, len(candidates), len(neighbors_unobs) ))
		# unobs = nodes.difference(obs_nodes)
		# if len(unobs) == 0:
		# 	unobs = nodes.intersection(self._sample['nodes']['open'])
		# 	print('		-- none --', len(unobs), len(nodes) )
		# 	if len(unobs) == 0:
		# 		unobs = self._sample['nodes']['open']

		return random.choice(list(candidates))

	def _sample_open_node_bak(self, candidate_list):
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		open_nodes = self._sample['nodes']['open']

		# deg_1 = self._sample_graph.degree(self._sample['nodes']['close'])
		# deg_2 = self._sample_graph.degree()
		# _mylib.degreeHist_2([deg_1.values(), deg_2.values()], save=True, legend=['close', 'all'])

		# Randomly pick some fraction of open nodes
		k = int((self._k / 100.) * len(open_nodes))
		open_nodes_sample = random.sample(open_nodes, k)

		self._tmp = len(open_nodes_sample)
		print(' -- Pick nodes {} / {} actual: {}'.format(k, len(open_nodes), len(open_nodes_sample)))

		# Find the best node with max unobseved degree
		best_node = self._query_max_unobs(open_nodes_sample)

		return best_node

	def _query_max_unobs(self, candidate_list):

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].update(candidate_list)
		obs_nodes = set(self._sample['nodes']['open']).union(set(self._sample['nodes']['close']))

		score_close = {}
		score_open = {}
		score_unobs = {}

		for node in set(candidate_list):
			nodes, edges, c2 = self._query.neighbors(node)

			score_close[node] = 1. * len(set(self._sample['nodes']['close']).intersection(set(nodes))) / len(nodes)
			score_open[node] = 1. * len(set(self._sample['nodes']['open']).intersection(set(nodes))) / len(nodes)
			score_unobs[node] = len(set(nodes) - obs_nodes)

			print(' \t\tNode: {} \tclose: {},\topen {},\tunobs {} \t {} \t {}'.format(node, score_close[node], score_open[node], score_unobs[node], self._cost, c2))

			self._increment_cost(c2)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			if self._cost > self._budget:
				break

		self._updateSample(sub_sample)

		#avg_new_nodes = np.mean(np.array(score_unobs.values()))
		idx = np.where(np.array(score_unobs.values()) == 0)[0]

		# TODO: Just use same variable name, it is a prob of not being 0
		avg_new_nodes = 1 - (1.*len(idx) / len(score_unobs))
		avg_close = np.mean(np.array(score_close.values()))
		avg_open = np.mean(np.array(score_open.values()))

		self._avg['unobs'].append(avg_new_nodes)
		self._avg['close'].append(avg_close)
		self._avg['open'].append(avg_open)

		best_node = _mylib.sortDictByValues(score_unobs,reverse=True)[0][0]
		best_val = _mylib.sortDictByValues(score_unobs,reverse=True)[0][1]

		if best_val != 0:
			nodes, edges, c2 = self._query.neighbors(best_node)
			tmp = set(nodes) - obs_nodes
		else:
			print('		-- switch -- ')
			best_node = _mylib.sortDictByValues(score_open, reverse=True)[0][0]
			nodes, edges, c2 = self._query.neighbors(best_node)
			tmp = (set(self._sample['nodes']['open']).intersection(set(nodes)))

		print('		==> Pick node {}: {} {} <== AVG new nodes: {} close : {} open : {}'.format(best_node, best_val, len(tmp),
																							   avg_new_nodes, avg_close,
																							   avg_open))

		ret = random.choice(list(tmp))

		return ret

	def _expansion_hubs_multi_nb(self, candidate_list, sample_g=None):
		if sample_g is None:
			sample_g = nx.Graph()
			sample_g.add_edges_from(self._sample['edges'])
			print(" * ", sample_g.number_of_edges())

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		# Graph with 1-deg nodes removed.
		sample_G_induced = _mylib.remove_node_with_deg(sample_g)
		partition = community.best_partition(sample_G_induced)
		mod = community.modularity(partition, sample_G_induced)
		cc_all = nx.clustering(sample_G_induced)

		# TODO: check these two
		#nodes_candidate = self._get_hubs_neighbors_as_candidate(partition, sample_G_induced, sample_g)
		nodes_candidate = set(sample_G_induced.nodes()).intersection(set(self._sample['nodes']['open']))
		cc = nx.clustering(sample_g, nodes_candidate)
		cc_g = _mylib.remove_one_values(cc)

		if len(cc_g) == 0:
			print('	No zero cc - full list {} {}'.format(len(cc), len(nodes_candidate)))
			cc_g = cc

		# TODO: value of k should be well-picked
		#k = int(.5 * (len(cc_g)))
		k = 50
		final_list = self._pick_nodes(cc_g, sample_G_induced, partition,k=k)
		if len(final_list) == 0:
			final_list = cc_g
		print('	Candidate List ', len(final_list))
		best_node = self._estimate_from_cheaper_layer(final_list, sample_g, sample_G_induced, partition)

		return best_node

	def _make_all_queries(self, candidate_list, sample_g, sample_G_induced, partition):
		if len(candidate_list) == 1:
			return candidate_list[0]

		best_unobs_count = 0
		best_node = None
		for node in candidate_list:
			# Query the neighbors of current
			nodes, edges, c2 = self._query.neighbors(node)


			self._increment_cost(c2)
			#self._cost += c2

			unobs_nodes_count = len(set(nodes) - set(sample_g.nodes()))

			if unobs_nodes_count > best_unobs_count:
				best_node = node
				best_unobs_count = unobs_nodes_count

		# If there is no best node found
		if best_node == None:
			best_node = random.choice(candidate_list)

		#print(' Completed {} '.format(best_unobs_count))
		return best_node

	def _estimate_from_cheaper_layer(self, candidate_list, sample_g, sample_G_induced, partition):
		if len(candidate_list) == 1:
			return candidate_list[0]
		elif len(candidate_list) == 0:
			return random.choice(list(self._sample['nodes']['open']))

		unobs_nodes_count_list = {}
		score_list = {}
		sum_cost = 0.0
		for node in candidate_list:
			nodes, edges, c2 = self._query_cheap.neighbors(node)

			self._increment_cost(c2)

			sum_cost += c2

			obs_nodes = sample_g.nodes()

			if len(nodes) != 0:
				unobs_nodes_count = len(set(nodes) - set(obs_nodes))
				unobs_nodes_count_list[node] = unobs_nodes_count
				p_nodes = sample_G_induced.neighbors(node)
				intersect = set(nodes).intersection(set(p_nodes))
				#score_list[node] = len(intersect) / len(nodes)
				if len(intersect) == 0:
					score_list[node] = 0

				#print(score_list[node], unobs_nodes_count)


				# obs = len( set(nodes).intersection(set(obs_nodes)) )
				#
				# score =  1.* obs / len(nodes)
				# print(obs, score)
				#
				# score_list[node] = score

		#print(score_list)
		#best_node =  _mylib.sortDictByValues(score_list)[0][0]



		if len(unobs_nodes_count_list) == 0:
			print('		Bad luck >< .. list is empty')
			return random.choice(candidate_list)

		# zero_m = score_list.keys()
		# val_zero = [unobs_nodes_count_list[x] for x in zero_m]
		# max_val = max(val_zero)
		# l = _mylib.get_members_from_com(max_val, unobs_nodes_count_list)
		# ll = list(set(zero_m).intersection(l))
		# best_node = random.choice(ll)

		max_val = max(unobs_nodes_count_list.values())
		l = _mylib.get_members_from_com(max_val,unobs_nodes_count_list)
		best_node = random.choice(l)


		print(' Cost spent in cheaper layer = {} -- Unobs: {} ^ {}'.format(sum_cost, max_val, len(l)))

		return best_node

	def _pick_nodes(self, d, graph, partition,k=5):
		ret = []

		score = {}
		for n in d.keys():
			nbs = graph.neighbors(n)
			p_set = set()

			for nb in nbs:
				p = partition[nb]
				p_set.add(p)
			score[n] = 1.*len(p_set) / len(nbs)

		score = _mylib.sortDictByValues(score, reverse=True)

		ret = [t[0] for t in score[:k]]
		return ret

	def _pick_nodes2(self, d, graph, partition, k=5):
		ret = []

		score = {}
		for n in d.keys():
			nbs = graph.neighbors(n)
			p_set = set()

			for nb in nbs:
				p = partition[nb]
				p_set.add(p)
			score[n] = 1. * len(p_set) / len(partition.values())

		score = _mylib.sortDictByValues(score, reverse=True)

		ret = [t[0] for t in score[:k]]

		return ret

	def _expansion_hub(self,candidate_list, sample_g=None):
		if sample_g is None:
			print('Random - expansion')
			return self._expansion_random(candidate_list)

		# Graph with 1-deg nodes removed.
		sample_G_induced = _mylib.remove_node_with_deg(sample_g)
		partition = community.best_partition(sample_G_induced)
		mod = community.modularity(partition, sample_G_induced)


		candidates = self._get_hubs_as_candidate(partition, sample_G_induced, sample_g)

		close_nodes = self._sample['nodes']['close']
		t = set(candidates) - close_nodes

	def _get_hubs_as_candidate(self,partition,sample_G_induced,sample_g):
		candidate = []
		for p in set(partition.values()):
			members = _mylib.get_members_from_com(p, partition)
			# Construct community graph - com_graph
			com_graph = sample_G_induced.subgraph(members)

			# TODO: Centrality or degree, should consider only particular community
			deg_com = nx.degree_centrality(com_graph)
			# deg_com = com_graph.degree(members)

			# Pick 10% of nodes that has the highest degree in the graph
			k_percent = self._k / 100.
			K_HUBS = int(k_percent * len(members))
			print("Hubs {} - {}".format(p,K_HUBS))
			deg_com = _mylib.sortDictByValues(deg_com, reverse=True)
			top_k_deg = [seq[0] for seq in deg_com[:K_HUBS]]
			candidate += top_k_deg


		return candidate

	def _get_dispersion_candidate_from_partition(self, partition,sample_G_induced,sample_g,return_min=True):
		dispersion_candidate = []

		p_degree = []
		for p in set(partition.values()):
			members = _mylib.get_members_from_com(p, partition)

			# Construct community graph - `com_graph`
			com_graph = sample_G_induced.subgraph(members)

			# TODO: Centrality or degree, should consider only particular community
			deg_com = nx.degree_centrality(com_graph)


			# Pick 10% of nodes that has the highest degree in the graph
			k_percent = self._k / 100.
			K_HUBS = int(k_percent*len(members))

			if K_HUBS == 0:
				K_HUBS = 1
			print('	{} is picked {} hubs'.format(p, K_HUBS))

			deg_com = _mylib.sortDictByValues(deg_com, reverse=True)
			top_k_deg = [seq[0] for seq in deg_com[:K_HUBS]]

			for hub in top_k_deg:
				nb = sample_G_induced.neighbors(hub)
				nb_t = self._sample['nodes']['close'] - set(nb)
				#print(nb)
				ego_g = nx.ego_graph(sample_G_induced, hub)
				ego_g.remove_nodes_from(nb_t)

				#_mylib.draw_com(ego_g, ego_g.degree())

				# Calculate dispersion starting from a selected node
				dispersion = nx.dispersion(sample_G_induced, hub)

				# Remove all closed nodes
				dispersion = _mylib.remove_entries_from_dict(self._sample['nodes']['close'], dispersion)
				dispersion = _mylib.sortDictByValues(dispersion, reverse=True)

				if return_min:
					dispersion_candidate += dispersion
				elif not return_min and len(dispersion) != 0:
					dispersion_candidate.append(dispersion[0])



		# return zero dispersion if return_min=True
		if return_min:
			non_zero_l, dispersion_candidate = _mylib.remove_zero_values(dispersion_candidate)
			print('	[Disp] Filter out - {} {}'.format(len(non_zero_l), len(dispersion_candidate) ))

		return _mylib.sort_tuple_list(dispersion_candidate)

	def _get_dispersion_candidate(self, partition, sample_G_induced, sample_g, return_min=True):
		deg_cen = nx.degree_centrality(sample_G_induced)
		deg_cen = _mylib.sortDictByValues(deg_cen, reverse=True)

		nodes_count = sample_G_induced.number_of_nodes()

		k_percent = self._k / 100.
		K_HUBS = int(k_percent * len(members))
		top_k_cen = deg_cen[:K_HUBS]

		dispersion_candidate = []
		for hub in top_k_cen:

			# Calculate dispersion starting from a selected node
			dispersion = nx.dispersion(sample_G_induced, hub[0])
			# Remove all closed nodes from the list
			dispersion = _mylib.remove_entries_from_dict(self._sample['nodes']['close'], dispersion)
			dispersion = _mylib.sortDictByValues(dispersion, reverse=True)
			dispersion_candidate += dispersion

			if return_min:
				dispersion_candidate += dispersion
			elif not return_min and len(dispersion) != 0:
				dispersion_candidate.append(dispersion[0])

		# return zero dispersion if return_min=True
		if return_min:
			non_zero_l, dispersion_candidate = _mylib.remove_zero_values(dispersion_candidate)


		return _mylib.sort_tuple_list(dispersion_candidate)

	def _get_hubs_neighbors_as_candidate(self, partition, sample_G_induced, sample_g, return_min=True):
		dispersion_candidate = []

		open_node = set()
		for p in set(partition.values()):
			members = _mylib.get_members_from_com(p, partition)

			# Construct community graph - `com_graph`
			com_graph = sample_G_induced.subgraph(members)
			# TODO: Centrality or degree, should consider only particular community
			deg_com = nx.degree_centrality(com_graph)

			# Pick 10% of nodes that has the highest degree in the graph
			k_percent = self._k / 100.
			K_HUBS = int(k_percent * len(members))

			if K_HUBS == 0:
				K_HUBS = 1
			#print('	Com: {} is picked {} hubs'.format(p, K_HUBS))

			deg_com = _mylib.sortDictByValues(deg_com, reverse=True)
			top_k_deg = [seq[0] for seq in deg_com[:K_HUBS]]

			for hub in top_k_deg:
				nb = sample_G_induced.neighbors(hub)
				open_nb = set(nb).difference(self._sample['nodes']['close'])
				#print('		# open nodes {} / {} '.format(len(open_nb), len(nb)))

				open_node.update(open_nb)

		return list(open_node)

	def _oracle_max_unobs_neigbors(self, candidate_list, sample_g):
		best_node, c = self._oracle.get_max_unobs_nodes(candidate_list, sample_g.nodes())

		self._increment_cost(c)
		#if self._isCost:
			#self._cost += c


		# cc = nx.clustering(self._oracle.graph, nodes_candidate)
		# cc_sorted = _mylib.sortDictByValues(cc,reverse=True)
		#
		# node_freq = {}
		# for node_t in candidate_list:
		# 	node = node_t[0]
		# 	node_freq[node] = node_freq.get(node,0) + 1
		#
		# for node_t in cc_sorted:
		# 	node = node_t[0]
		# 	node_cc = node_t[1]
		#
		# 	if node == best_node:
		# 		print('{} {} *'.format(node_cc, node_freq[node]))
		# 	else:
		# 		print('{} {} '.format(node_cc, node_freq[node]))



		# best_count = 0
		# best_node = None
		# for node_t in nodes_track:
		# 	node = node_t[0]
		# 	freq = node_t[1]
		#
		# 	# Assume we ask the Oracle about number of unobserved degree of nodes.
		# 	nodes, edges, c2 = self._query.neighbors(node)
		# 	# Focus on the unobserved NODE instead of unobserved DEGREE
		# 	unobserved_nodes_count = len(set(nodes) - set(sample_g.nodes()))
		#
		# 	if unobserved_nodes_count > best_count or best_count == 0:
		# 		best_count = unobserved_nodes_count
		# 		best_node = node

		return best_node

	def _expansion(self, candidate_list):
		"""
		Run the expansion step

		Args:
			candidate_list (list[str]) -- List of expansion candidates
		Return:
			str -- The id of the node to perform densification on
		"""
		sample_g = self._sample_graph
		sample_g_in = sample_g.subgraph(self._sample['nodes']['close'])



		obs_nodes = list(self._sample['nodes']['close'].union(self._sample['nodes']['open']))
		# TODO: if want to balance distance, call this
		#center_node = nx.center(sample_g_in)[0]
		#current, c = self._oracle.expansion(candidate_list, obs_nodes, center_node)
		current, c = self._oracle.expansion(candidate_list, obs_nodes, "")

		self._increment_cost(c)

		return current

	def _expansion_old(self, candidate_list):
		"""
		Run the expansion step

		Args:
			candidate_list (list[str]) -- List of expansion candidates
		Return:
			str -- The id of the node to perform densification on
		"""
		current, c = self._oracle.expansion_old(candidate_list, \
											list(self._sample['nodes']['close'].union(self._sample['nodes']['open'])))

		self._increment_cost(c)

		return current

	def _densification_oracle(self, candidate):
		"""
		Run the densification steps

		Args:
			candidate (str) -- The id of the node to start densification on
		Return:
			list[str] -- List of candidate nodes for expansion
		"""

		# If the candidate is not in the sample, add it
		if candidate not in self._sample_graph.nodes():
			self._sample_graph.add_node(candidate)

		# Initialize a new sub sample
		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}
		sub_sample['nodes']['open'].add(candidate)

		# Initialize densification and expansion scores
		# score_den = self._scoreDen(sub_sample)
		score_den = self._scoreDen_test(sub_sample)
		score_exp = self._scoreExp(sub_sample)

		prev_score = score_den
		score_change = 1.
		score_list = []
		RANGE = 3
		THRESHOLD = 1.
		isConverge = False

		# Perform densification until one of the conditions is met:
		# 	1. Densification score is less than the expansion score
		# 	2. Budget allocated has run out
		# 	3. There are no more open nodes
		# TODO: Densification switch criteria
		# while score_exp < score_den and self._cost < self._budget\
		#  and len(sub_sample['nodes']['open']) > 0:

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			# Get the list of nodes to perform densification on
			den_nodes = self._getDenNodes(sub_sample['nodes']['open'])
			#den_nodes = sub_sample['nodes']['open']

			# Get the node to densify on from the Oracle
			observed_nodes = list(self._sample_graph.nodes())

			current, c1 = self._oracle.densification(den_nodes, observed_nodes)
			current_node_obs_deg = self._sample_graph.degree(current)

			# Query the neighbors of current
			nodes, edges, c2 = self._query.neighbors(current)

			# Update the densification and expansion scores
			score_den = self._scoreDen_test(sub_sample, nodes, current_node_obs_deg, score_den)
			score_exp = self._scoreExp(sub_sample)
			score_list.append(score_den)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			self._increment_cost(c2)

			print(' [oracle] score_den: {} current_deg: {} '.format(score_den, current_node_obs_deg))

			if score_den < 0.1:
				break

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

		# Return list of potential expansion nodes
		return self._getExpNodes()

	def _densification(self, candidate):
		"""
		Run the densification steps

		Args:
			candidate (str) -- The id of the node to start densification on
		Return:
			list[str] -- List of candidate nodes for expansion
		"""

		# If the candidate is not in the sample, add it
		if candidate not in self._sample_graph.nodes():
			self._sample_graph.add_node(candidate)

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(candidate)

		# Initialize densification and expansion scores
		#score_den = self._scoreDen(sub_sample)
		score_den = self._scoreDen_test(sub_sample)
		score_exp = self._scoreExp(sub_sample)

		# # Initialize unobs list
		# self._den_unobs = []

		prev_score = score_den
		score_change = 1.
		score_list = []
		THRESHOLD = 1.
		isConverge = False

		# Perform densification until one of the conditions is met:
		# 	1. Densification score is less than the expansion score
		# 	2. Budget allocated has run out
		# 	3. There are no more open nodes
		# TODO: Densification switch criteria
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			# Get the list of nodes to perform densification on
			#den_nodes = self._getDenNodes(sub_sample['nodes']['open'])
			# TODO: Nodes for densify should be filter out ?
			den_nodes = sub_sample['nodes']['open']

			# degree_observed = sample_G.degree(den_nodes)
			degree_observed = self._sample_graph.degree(den_nodes)

			if len(den_nodes) != 1:
				degree_observed_sorted = _mylib.sortDictByValues(degree_observed,reverse=True)
				current = degree_observed_sorted[0][0]
				current_node_obs_deg = degree_observed_sorted[0][1]
			else:
				print('no den nodes')
				current = list(den_nodes)[0]
				current_node_obs_deg = degree_observed[current]

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current)
			#sample_G.add_edges_from(edges)

			# TODO: Densification score, to be changed.
			# Update the densification and expansion scores
			score_den = self._scoreDen_test(sub_sample,nodes,current_node_obs_deg,score_den)
			#score_den = self._scoreDen(sub_sample, nodes, score_den)
			score_exp = self._scoreExp(sub_sample, score_exp)

			# Store the densification and expansion scores
			self._score_den_list.append(score_den)
			self._score_exp_list.append(score_exp)
			self._densi_count += 1

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			# TODO: just a dummy statement for MOD method
			if self._exp_type != 'mod' and score_den < 0.1:
					break

			print('score_den: {}'.format(score_den))

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

		# Return list of potential expansion nodes
		return self._getExpNodes()

	def _densification_max_score(self, candidate):
		"""
		Run the densification steps

		Args:
			candidate (str) -- The id of the node to start densification on
		Return:
			list[str] -- List of candidate nodes for expansion
		"""
		current_node = candidate

		# If the candidate is not in the sample, add it
		if candidate not in self._sample_graph.nodes():
			self._sample_graph.add_node(candidate)

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(candidate)

		# Initialize densification and expansion scores
		# score_den = self._scoreDen(sub_sample)
		score_den = self._scoreDen_test(sub_sample)

		prev_score = score_den


		print('Start Den - ', candidate)
		# Perform densification until one of the conditions is met:
		# 	1. Densification score is less than the expansion score
		# 	2. Budget allocated has run out
		# 	3. There are no more open nodes
		# TODO: Densification switch criteria
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			current_node_obs_deg = self._sample_graph.degree(current_node)
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)


			# Update the densification and expansion scores
			score_den = self._scoreDen_test(sub_sample, nodes, current_node_obs_deg, score_den)

			print('Current node', current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			# TODO: just a dummy statement for MOD method
			if score_den < 0.1:
				break

			print('[max-score] score_den: {}'.format(score_den))

			# Candidate nodes are the (open) neighbors of current node
			candidates = list(
				set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))

			while len(candidates) == 0:
				current_node = random.choice(list(nodes))
				# Query the neighbors of current
				nodes, edges, c = self._query.neighbors(current_node)
				# Candidate nodes are the (open) neighbors of current node
				candidates = list(
					set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
				print("RW: getting stuck")

			current_node = random.choice(candidates)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

		# Return list of potential expansion nodes
		return self._getExpNodes()

	def random_sampling(self):

		current = starting_node

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		# TODO: Densification switch criteria
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			den_nodes = sub_sample['nodes']['open']

			# Randomly pick node
			current = random.choice(list(den_nodes))

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _score_R(self, sub_sample):
		g = self._sample_graph
		closed_nodes = self._sample['nodes']['close'].union(sub_sample['nodes']['close'])
		o = self._sample['nodes']['open'].union(sub_sample['nodes']['open'])
		open_nodes = set(o).difference(closed_nodes)

		b_nodes = self._get_boundary_nodes(closed_nodes, open_nodes)

		t_nodes = set()
		b_edges = list(g.edges_iter(list(b_nodes)))
		for e in b_edges:
			t_nodes.add(e[1])

		try:
			r_score = len(set(t_nodes).intersection(set(closed_nodes))) / len(b_edges)
		except ZeroDivisionError:
			r_score = 1.

		#print(' Border: {} {} Closed: {} Open: {} T: {} score: {}'.format(len(b_nodes), len(b_edges), len(closed_nodes), len(open_nodes), len(t_nodes), r_score))

		return r_score

	def _get_boundary_nodes(self,closed_nodes, open_nodes):
		g = self._sample_graph
		b_nodes = set()
		for c_n in closed_nodes:
			nbs = g.neighbors(c_n)
			if len(set(nbs).intersection(set(open_nodes))) != 0:
				b_nodes.add(c_n)
		return b_nodes

	def _score(self, sub_sample, nodes=None, prev=0):
		# This should be for only the start of the densification
		if nodes is None:
			return np.inf

		# The new nodes; Nodes neithen in sample and sub sample
		new_nodes = nodes.difference(sub_sample['nodes']['close']) \
			.difference(sub_sample['nodes']['open']) \
			.difference(self._sample['nodes']['open']).difference(self._sample['nodes']['close'])

		#print('	Unobs {} / {} '.format(len(new_nodes), len(nodes)))

		med = np.median(np.array(self._new_nodes))

		# Calculate the densification score
		#score = 1.*len(new_nodes) / (len(sub_sample['nodes']['close']) + 1)
		score = 1.*len(new_nodes) / med

		print('score {} \t unobs {} \t med {} \t '.format(score,len(new_nodes),med))

		# Store number of new nodes for logging later
		self._new_nodes.append(len(new_nodes))

		if np.isfinite(prev):
			return (0.5 * prev) + score
		else:
			return score

	def _scoreDen_test(self, sub_sample, nodes=None, obs_deg=0, prev=0):
		"""
		Calculate the densification score

		Args:
			sub_sample(dict) -- Dict of the subsample
			nodes (list[str]) -- Unobserved nodes from last densification step (default: None)
			prev (float) -- The previous score (default: 0)
		Return:
			float -- The densification score
		"""

		# This should be for only the start of the densification
		if nodes is None:
			return 1.
			#return np.inf

		# The new nodes; Nodes neithen in sample and sub sample
		new_nodes = nodes.difference(sub_sample['nodes']['close']) \
			.difference(sub_sample['nodes']['open']) \
			.difference(self._sample['nodes']['open']).difference(self._sample['nodes']['close'])

		print('	Unobs {} / {} '.format(len(new_nodes), len(nodes)))

		# TODO: score function
		# Calculate the densification score
		try:
			score = len(new_nodes) / (len(nodes) - obs_deg)
		except ZeroDivisionError:
			score = 0.

		print(' new nodes : excess deg  - {}/{} '.format(len(new_nodes), (len(nodes) - obs_deg)))

		# Store number of new nodes for logging later
		self._new_nodes.append(len(new_nodes))

		if np.isfinite(prev):
			return (0.5 * prev) + (self._wt_den * score)
		else:
			return self._wt_den * score

	def _scoreDen(self, sub_sample, nodes=None, prev=0):
		"""
		Calculate the densification score

		Args:
			sub_sample(dict) -- Dict of the subsample
			nodes (list[str]) -- Unobserved nodes from last densification step (default: None)
			prev (float) -- The previous score (default: 0)
		Return:
			float -- The densification score
		"""

		# This should be for only the start of the densification
		if nodes is None:
			return np.inf

		# The new nodes; Nodes neithen in sample and sub sample
		new_nodes = nodes.difference(sub_sample['nodes']['close'])\
		 .difference(sub_sample['nodes']['open'])\
		 .difference(self._sample['nodes']['open']).difference(self._sample['nodes']['close'])

		#print('	Unobs {} / {} '.format(len(new_nodes), len(nodes)))

		# TODO: score function
		# Calculate the densification score
		score = len(new_nodes)/(len(sub_sample['nodes']['close']) + 1)
		#new_node_norm = len(new_nodes) / (len(nodes) - self._node_picked_obs_deg[-1:][0])

		#score = new_node_norm


		# Store number of new nodes for logging later
		self._new_nodes.append(len(new_nodes))

		if len(self._cumulative_new_nodes) == 0:
			self._cumulative_new_nodes.append(len(new_nodes))
		else:
			c_new_nodes = self._cumulative_new_nodes[-1]
			self._cumulative_new_nodes.append(c_new_nodes + len(new_nodes))


		if np.isfinite(prev):
			return 0.5 * prev + self._wt_den * score
		else:
			return self._wt_den * score

	def _scoreExp(self, sub_sample, prev=0):
		"""
		Calculate the expansion score

		Args:
			sub_sample(dict) -- The sub sample from current densification
			prev (float) -- The previous expansion score
		"""

		# Get the edges between open and close nodes in the current sub sample
		edges = set()
		for e in sub_sample['edges']:
			if (e[0] in sub_sample['nodes']['close']\
			 and e[1] in sub_sample['nodes']['open'])\
			 or (e[1] in sub_sample['nodes']['close']\
			 and e[0] in sub_sample['nodes']['open']):
				edges.add(e)

		# Calculate the expansion score
		score = len(edges)/(len(sub_sample['nodes']['open']) + 1)

		return 0.5 * prev + self._wt_exp * score

	def _updateSample(self, sub_sample):
		"""
		Update the sample with the sub sample

		Args:
			sub_sample (dict) -- The sub sample dict
		"""
		self._sample['edges'].update(sub_sample['edges'])
		self._sample['nodes']['close'].update(sub_sample['nodes']['close'])
		self._sample['nodes']['open'] = self._sample['nodes']['open'].difference(\
		 sub_sample['nodes']['close'])
		self._sample['nodes']['open'].update(sub_sample['nodes']['open'])


		nodes_count = self._sample_graph.number_of_nodes()
		# TODO: Calculate mean and median of close nodes degree, MIGHT CHANGE !
		#degree = self._sample_graph.degree().values()
		degree = self._sample_graph.degree(self._sample['nodes']['close']).values()
		self._avg_deg = np.mean(np.array(degree))
		self._med_deg = np.median(np.array(degree))
		# print( " Degree avg: {} , med: {}".format(self._avg_deg, self._med_deg))

	def _updateSubSample(self, sub_sample, nodes, edges, candidate):
		"""
		Update the sub sample with new nodes aned edges

		Args:
			sub_sample (dict) -- The sub sample to update
			nodes (list[str]) -- The open nodes
			edges (list[(str,str)]) -- The new edges
			candidate (str) -- The new open node
		Return:
			dict -- The updated sub sample
		"""
		try:

			sub_sample['edges'].update(edges)
			sub_sample['nodes']['close'].add(candidate)
			sub_sample['nodes']['open'].remove(candidate)
			sub_sample['nodes']['open'].update(\
				nodes.difference(sub_sample['nodes']['close'])\
				.difference(self._sample['nodes']['close']))
		except KeyError:
			print('		Err:')


		return sub_sample

	def _bfs(self):

		"""
		Collect the initial nodes through bfs

		Args:
			None
		Return:
			None
		"""

		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}

		current = starting_node

		sub_sample['nodes']['open'].add(current)
		queue = [current]

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._budget and len(queue) > 0:
			# Select the first node from queue
			current = queue[0]

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c = self._query.neighbors(current)
			self._increment_cost(c)

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Remove the current node from queue
			queue.remove(current)

			# Update queue
			nodes = nodes.difference(sub_sample['nodes']['close'])
			nodes = nodes.difference(sub_sample['nodes']['open'])
			queue += list(nodes)
			queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

	def _snowball_sampling(self):

		if starting_node == -1:
			print('Please check the starting node')
			return

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}

		print('start at ', starting_node)
		current = starting_node

		sub_sample['nodes']['open'].add(current)
		queue = [current]

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._budget and len(queue) > 0:
			# Select the first node from queue
			current = queue[0]

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c = self._query.neighbors(current)
			self._increment_cost(c)

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Remove the current node from queue
			queue.remove(current)

			# Update queue
			nodes = nodes.difference(sub_sample['nodes']['close'])
			nodes = nodes.difference(sub_sample['nodes']['open'])

			node_to_add = random.sample(nodes, int(.5*len(nodes)))
			queue += list(node_to_add)
			queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

	def _random_walk(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost
			self._increment_cost(c)

			# Candidate nodes are the (open) neighbors of current node
			candidates = list(set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))

			while len(candidates) == 0:
				current_node = random.choice(list(nodes))
				# Query the neighbors of current
				nodes, edges, c = self._query.neighbors(current_node)
				# Candidate nodes are the (open) neighbors of current node
				candidates = list(set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
				print("RW: getting stuck")

			current_node = random.choice(candidates)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _random_walk_max(self):
		current_node = random.choice(list(self._sample['nodes']['open']))

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost
			self._increment_cost(c)

			# Candidate nodes are the (open) neighbors of current node
			candidates = list(
				set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))

			while len(candidates) == 0:
				current_node = self._choose_next_node(list(nodes))
				# Query the neighbors of current
				nodes, edges, c = self._query.neighbors(current_node)
				# Candidate nodes are the (open) neighbors of current node
				candidates = list(
					set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
				print("RW: getting stuck")

			#current_node = random.choice(candidates)
			current_node = self._choose_next_node(candidates)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _choose_next_node(self, candidates):
		avg_cc = nx.average_clustering(self._sample_graph)
		deg_cand = self._sample_graph.degree(candidates)

		P_max = avg_cc
		rand = random.uniform(0,1)

		if rand <= P_max:
			selected_node = self._pick_max_score(deg_cand)
		else:
			selected_node = random.choice(candidates)

		return selected_node





	def _test(self):
		candidates = self._sample['nodes']['open']
		degree_observed = self._sample_graph.degree(candidates)
		degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
		current_node = degree_observed_sorted[0][0]

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		P_MOD = 0.5

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			#cc_node = nx.clustering(self._sample_graph, current_node)
			average_cc = nx.average_clustering(self._sample_graph)
			try:
				average_cc_close = nx.average_clustering(self._sample_graph.subgraph(close_n))
			except ZeroDivisionError:
				average_cc_close = 0.

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(
					self._sample['nodes']['close']))

			r = random.uniform(0,1)

			if r < P_MOD:
				current_node = random.choice(candidates)
			else:
				degree_observed = self._sample_graph.degree(candidates)
				degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
				current_node = degree_observed_sorted[0][0]

			print(' [test] current cc: {} {}'.format(average_cc, average_cc_close))

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_obs_deg(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			# cc_node = nx.clustering(self._sample_graph, current_node)
			#self._line_1.append(cc)
			# self._line_2.append(degree_observed_sorted[0][1])

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))

			degree_observed = self._sample_graph.degree(candidates)
			degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
			current_node = degree_observed_sorted[0][0]



		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_score(self):
		candidates = self._sample['nodes']['open']

		degree_observed = self._sample_graph.degree(candidates)
		# cc_observed = nx.clustering(self._sample_graph, candidates)
		cc_observed = nx.clustering(self._oracle.graph, candidates)
		cc_avg = nx.average_clustering(self._sample_graph)

		score = self._cal_score(degree_observed, cc_observed, cc_avg)
		current_node = self._pick_max_score(score)

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(
					self._sample['nodes']['close']))

			degree_observed = self._sample_graph.degree(candidates)
			cc_observed = nx.clustering(self._sample_graph, candidates)
			cc_avg = nx.average_clustering(self._sample_graph)

			score = self._cal_score(degree_observed, cc_observed, cc_avg)
			current_node = self._pick_max_score(score)
			#score_sorted = _mylib.sortDictByValues(score, reverse=True)

			#current_node = score_sorted[0][0]

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _cal_score(self, degrees,cc, avg_cc):
		d = {}
		max_deg = max(degrees.values())
		avg_deg = np.mean(np.array(degrees.values()))
		print(' cal score', max_deg, avg_deg)
		for k,v in degrees.iteritems():
			#if degrees[k] != 1:
			#d[k] = (degrees[k]/avg_deg) * (1/(cc[k]+0.001))
			d[k] = (degrees[k]/max_deg) * (1 - cc[k])

		return d


	def _pick_max_score(self, score):
		max_val = max(score.values())
		np_score = np.array(score.values())
		max_idx = np.argmax(np_score)
		node = score.keys()[max_idx]
		#print(np_score)
		print(' max-score pick:', score[node])
		return node



	def _learn_model(self):
		candidates = self._sample['nodes']['open']
		degree_observed = self._sample_graph.degree(candidates)
		degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
		current_node = degree_observed_sorted[0][0]

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)
			self._get_training(sub_sample, nodes, edges, current_node)
			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(
					self._sample['nodes']['close']))
			# Start picking a new node to query
			if self._cost <= 50:
				degree_observed = self._sample_graph.degree(candidates)
				degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
				current_node = degree_observed_sorted[0][0]
			else:
				y = np.array(self._Y)
				#cut_off = np.median(self._sample_graph.degree(sub_sample['nodes']['close']).values())
				# cut_off = np.mean(y)
				#cut_off = 0.1

				#y[np.array(self._Y) >= cut_off] = 1
				#y[np.array(self._Y) < cut_off] = 0


				model = self._build_model(y)
				testing = self._get_testing(candidates)

				#print(np.shape(self._X))
				print(np.shape(np.array(testing)))


				#candidates = random.sample(candidates,5)
				#candidates_deg = self._sample_graph.degree(candidates)
				#candidates_cc = nx.clustering(self._sample_graph,candidates)

				#A = np.array([candidates_deg.values(), candidates_cc.values()]).transpose()
				#print(len(candidates))
				#print(A)

				y_predict = model.predict(testing)
				# print(y_predict)
				#
				max_val = (max(y_predict))
				print('max val', max_val)
				y_idx = np.where(y_predict == max_val)[0]

				#
				#
				# # y_idx = np.nonzero(y_predict)[0]
				# #
				if len(y_idx) != 0:
					pick = random.choice(y_idx)
					#print('pick index', pick, len(candidates))
					current_node = list(candidates)[pick]
				# else:
				# 	print('all zeros')
				# 	current_node = random.choice(candidates)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _build_model(self, y):
		np_X = np.array(self._X)

		logistic = linear_model.LogisticRegression()
		logistic.fit(np_X, y)

		return logistic

	def _getDenNodes(self, nodes):
		"""
		Generate a list of best densification nodes based on clustering coeff

		Only the number of nodes with count highest clustering
		coefficient are to be considered for densification

		Args:
			nodes(list([str])) -- Open nodes list
		Return:
			list[str] -- List of self._den_cut_off with highest clustering coeff
		"""
		if len(nodes) > self._den_cut_off:
			# Get clustering coefficients of the nodes
			cc = nx.clustering(self._sample_graph, nodes)
			# Sort nodes by clustering coefficient in descending order
			max_val = cc.values()
			candidates = _mylib.get_members_from_com(max_val,cc)

			if len(candidates) > self._den_cut_off:
				return random.sample(candidates, self._den_cut_off)
			else:
				return candidates
			# cc = sorted(cc, key=cc.get, reverse=True)
			# return cc[:self._den_cut_off]
		else:
			return list(nodes)

	def _getExpNodes(self):
		"""
		Generate a list of best expansion nodes based on clustering coeff

		Only the number of nubmer of nodes with count lowest clustering
		coefficient are to be considered for expansion

		Considers all the open nodes. Not just from latest subsample

		Args:
			None
		Return:
			list[str] -- The self._exp_cut_off nodes with lowest clustering coeff
		"""
		if len(self._sample['nodes']['open']) > 0:
			# Clustering coeff of the open nodes in the sample
			cc = nx.clustering(self._sample_graph, self._sample['nodes']['open'])
			# Sort the nodes by clustering coeff in ascending order
			cc = sorted(cc, key=cc.get, reverse=False)
			return cc[:self._exp_cut_off]
			#return cc
		else:
			print('	*No open nodes')
			return list(nodes)

	def _track_cost_spent_return(self):
		cur_return = len(self._sample['nodes']['close']) + len(self._sample['nodes']['open'])

		if len(self._cost_spent) == 0:
			prev_cost = 0
			prev_return = 0
		else:
			prev_cost = self._cost_spent[-1]
			prev_return = self._nodes_return[-1]

		cost_used = self._cost - prev_cost
		cost_used = self._cost - prev_cost
		node_gain = cur_return - prev_return

		self._nodes_return.append(node_gain)
		self._cost_spent.append(cost_used)

		#self._nodes_return.append(cur_return - prev_return)

		# self._cost_spent.append(self._cost - prev_cost)

		#self._cost_spent.append([self._stage, self._cost, cur_return])
		#self._nodes_return.append(cur_return)

	def _increment_cost(self, cost ):
		self._cost += cost


		if  int(self._cost) % self._log_interval == 0:
			obs_nodes = self._sample_graph.number_of_nodes()

			c = int(self._cost)

			self._track[c] = obs_nodes
			self._track_k.append(self._tmp)
			self._track_open.append(len(self._sample['nodes']['open']))


			nodes_count = self._sample_graph.number_of_nodes()
			edges_count = self._sample_graph.number_of_edges()

			self._line_1.append(nodes_count)
			self._line_2.append(edges_count)

	def generate(self):
			"""
			The main method that calls all the other methods
			"""
			#self._bfs()
			current_list = []

			sample_G = None
			is_break = False

			# Sample until budget runs out or thero are no more open nodes
			while self._cost < self._budget:

				# If there are no more nodes in current list, use open nodes from sample
				if len(current_list) < 1:
					current_list = list(self._sample['nodes']['open'])

				# Perform expansion
				self._stage = 'exp'
				if self._exp_type == 'oracle':
					current_list = self._getExpNodes()
					current = self._expansion(current_list)
					self._exp_count += 1
				elif self._exp_type == 'random-exp':
					current = self._expansion_random(self._sample['nodes']['open'])
					self._exp_count += 1
				elif self._exp_type == 'percentile-exp':
					current = self._sample_open_node(current_list)
					self._exp_count += 1




				# Perform densification
				self._stage = 'den'
				if self._exp_type == 'random':
					self.random_sampling()
				elif self._exp_type == 'rw':
					self._random_walk()
				elif self._exp_type == 'mod':
					self._max_obs_deg()
				elif self._exp_type == 'max-score':
					self._max_score()
				elif self._exp_type == 'oracle':
					current_list = self._densification_oracle(current)
				elif self._exp_type == 'sb':
					self._snowball_sampling()
				elif self._exp_type == 'bfs':
					self._bfs()
				else:
					# current_list = self._densification(current)
					current_list = self._densification_max_score(current)

				self._densi_count += 1

				print('			Budget spent: {}/{}'.format(self._cost, self._budget))

			print('			Number of nodes \t Close: {} \t Open: {}'.format( \
				len(self._sample['nodes']['close']), \
				len(self._sample['nodes']['open'])))

			"""
			repititions = 0
			for x in self._oracle._communities_selected:
				repititions += self._oracle._communities_selected[x]
			repititions = repititions - len(self._oracle._communities_selected)
			print(self._oracle._communities_selected, len(self._oracle._communities_selected), repititions)
			"""

	# TODO: Normal function that runs with control budget
	def generate_bak(self):
		"""
		The main method that calls all the other methods
		"""
		self._bfs()
		current_list = []

		sample_G = None
		is_break = False

		# Sample until budget runs out or thero are no more open nodes
		while self._cost < self._budget \
				and len(self._sample['nodes']['open']) > 0:
			# If there are no more nodes in current list, use open nodes from sample
			if len(current_list) < 1:
				current_list = list(self._sample['nodes']['open'])

			cost_before = self._cost

			# Perform expansion
			self._stage = 'exp'
			if self._exp_type == 'oracle':
				current_list = self._getExpNodes()
				current = self._expansion(current_list)
			if self._exp_type == 'oracle_old':
				current_list = self._getExpNodes()
				current = self._expansion_old(current_list)
			elif self._exp_type == 'random-exp' or self._exp_type == 'random-samp':
				current = self._expansion_random(self._sample['nodes']['open'])
			elif self._exp_type == 'hyb':
				current_g = self._sample_graph
				sample_close = self._sample['nodes']['close']
				sample_open = self._sample['nodes']['open']
				deg_close = sorted(current_g.degree(sample_close).values())
				deg_open = sorted(current_g.degree(sample_open).values())

				# _mylib.plotLineGraph([deg_close, deg_open], legend=['close', 'open'])
				low_index = int(math.ceil(0.05 * len(deg_close)))
				avg_min_close = np.mean(np.array(deg_close[:low_index+1]))

				# mean_close = np.mean(np.array(deg_close))
				# med_close = np.median(np.array(deg_close))
				min_close = min(deg_close)
				# max_close = max(deg_close)

				mean_open = np.mean(np.array(deg_open))
				med_open = np.median(np.array(deg_open))
				min_open = min(deg_open)
				max_open = max(deg_open)


				print('Avg. min deg close: {} \t min close: {} \t Median deg open: {} '.format(avg_min_close, min_close, med_open))

				if avg_min_close <= med_open:
					print(' ## switch to random ##')
					current = self._expansion_random(self._sample['nodes']['open'])
				else:
					current = self._sample_open_node(current_list)
			elif self._exp_type == 'percentile-exp':
				current = self._sample_open_node(current_list)
			# elif self._exp_type == 'mod':
			# 	# Pick max obs nodes instead of perfroming Expansion
			# 	sample_G = self._sample_graph
			# 	degree = _mylib.sortDictByValues(sample_G.degree(self._sample['nodes']['open']), reverse=True)
			# 	print(" MOD degree {} {}".format(degree[0][0], degree[0][1]))
			# 	current = degree[0][0]

			cost_after = self._cost
			cost_spent = cost_after - cost_before
			self._one_cost += cost_spent
			self._exp_count += 1

			# Track
			self._track_cost_spent_return()

			# Perform densification
			self._stage = 'den'
			if self._exp_type == 'random-samp':
				current_list = self._densification_random(current)
			elif self._exp_type == 'rw':
				self._random_walk()
			elif self._exp_type == 'mod':
				self._max_obs_deg()
			elif self._exp_type == 'max-score':
				self._max_score()
			elif self._exp_type == 'oracle':
				current_list = self._densification_oracle(current)
			elif self._exp_type == 'test':
				self._random_walk_max()
			else:
				# current_list = self._densification(current)
				current_list = self._densification_max_score(current)
			# Track
			self._track_cost_spent_return()


			# current_g = self._sample_graph
			# sample_close = self._sample['nodes']['close']
			# sample_open = self._sample['nodes']['open']
			# deg_close = sorted(current_g.degree(sample_close).values())
			# deg_open = sorted(current_g.degree(sample_open).values())
			#
			# mean_close = np.mean(np.array(deg_close))
			# med_close = np.median(np.array(deg_close))
			# min_close = min(deg_close)
			# max_close = max(deg_close)
			#
			# mean_open = np.mean(np.array(deg_open))
			# med_open = np.median(np.array(deg_open))
			# min_open = min(deg_open)
			# max_open = max(deg_open)

			# print('C: {} \t {} \t {} \t {}'.format(max_close, mean_close, med_close, min_close))
			# print('O: {} \t {} \t {} \t {}'.format(max_open, mean_open, med_open, min_open))

			print('Budget spent: {}/{}'.format(self._cost, self._budget))


		print('Number of nodes \t Close: {} \t Open: {}'.format( \
			len(self._sample['nodes']['close']), \
			len(self._sample['nodes']['open'])))

		"""
		repititions = 0
		for x in self._oracle._communities_selected:
			repititions += self._oracle._communities_selected[x]
		repititions = repititions - len(self._oracle._communities_selected)
		print(self._oracle._communities_selected, len(self._oracle._communities_selected), repititions)
		"""

		# log.log_new_nodes(self._logfile, self._dataset, log_cost, log_new_nodes, self._budget, self._bfs_count)

		# log.writeUndirectedSingleLayer(self._logfile, self._dataset, \
		# 							   self._budget, self._bfs_count, self._oracle._cost_expansion, \
		# 							   self._oracle._cost_densification, self._exp_cut_off, self._den_cut_off, \
		# 							   self._sample, self._new_nodes, self._score_exp_list, self._score_den_list,
		# 							   self._nodes_observed_count)

	def ego_coverage(self):

		ego_nodes = []
		ego_edges = []
		tot_nodes = self._oracle.graph.number_of_nodes()
		tot_edges = self._oracle.graph.number_of_edges()
		for n in self._oracle.graph.nodes():
			sub_graph = nx.ego_graph(G, n)
			ego_nodes.append(sub_graph.number_of_nodes() / tot_nodes)
			ego_edges.append(sub_graph.number_of_edges() / tot_edges)

		log.log_ego_graph(sorted(ego_nodes, reverse=True), sorted(ego_edges, reverse=True))

def Logging(sample):
	# Keep the results in file
	track_sort = _mylib.sortDictByKeys(sample._track)
	cost_track = [x[0] for x in track_sort]
	obs_track = [x[1] for x in track_sort]


	log.log_new_nodes(log_file, dataset, type, obs_track, cost_track, budget, bfs_budget)

	print("---- DONE! ----")
	print("	# Exp: {}, # Den: {}".format(sample._exp_count, sample._densi_count))
	print('	Nodes in S: ', sample._sample_graph.number_of_nodes())
	#print('	Clustering Coeff =', nx.average_clustering(graph))
	print('-'*15)

def SaveToFile(results):
	log.save_to_file(log_file, results)


def Append_Log(sample, type):
	track_sort = _mylib.sortDictByKeys(sample._track)
	cost_track = [x[0] for x in track_sort]
	obs_track = [x[1] for x in track_sort]

	if type not in Log_result:
		Log_result[type] = obs_track
	else:
		Log_result[type] += (obs_track)

	return cost_track


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-task', help='Type of sampling', default='undirected_single')
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-budget', help='Total budget', type=int, default=1000)
	parser.add_argument('-bfs_budget', help='Bfs budget', type=int, default=5)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)
	parser.add_argument('-log', help='Log file', default='./log/')
	parser.add_argument('-experiment', help='# of experiment', default=10)
	parser.add_argument('-log_interval', help='# of budget interval for logging', type=int, default=10)
	parser.add_argument('-k', help='top k percent', type=int, default=5)
	parser.add_argument('-is_cost', help='take cost in account', type=bool, default=True)
	parser.add_argument('-mode', help='mode', type=int, default=1)
	parser.add_argument('-delimiter', help='csv delimiter', type=str, default=None)

	args = parser.parse_args()

	print(args)

	fname = args.fname
	budget = args.budget
	bfs_budget = args.bfs_budget
	dataset = args.dataset
	log_file = args.log
	k = args.k
	is_cost = args.is_cost
	log_interval = args.log_interval
	mode = args.mode
	delimeter = args.delimiter


	if mode == 1: exp_list = ['mod','rw','random','sb','bfs']
	# elif mode == 2: exp_list = ['oracle', 'random-exp', 'percentile-exp','mod','random-samp']
	# elif mode == 3: exp_list = ['percentile-exp']
	# elif mode == 4: exp_list = ['mod']
	# elif mode == 5: exp_list = ['percentile-exp']
	# elif mode == 6: exp_list = ['oracle','percentile-exp']
	# elif mode == 7: exp_list = ['oracle', 'max-score', 'mod','rw','random-samp']
	# elif mode == 8: exp_list = ['oracle', 'mod', 'rw', 'random-samp']
	# elif mode == 9: exp_list = ['sb']

	print(exp_list)
	Log_result = {}

	if args.task == 'undirected_single':
		G = nx.read_edgelist(fname, delimiter=delimeter)
		print('Original: # nodes', G.number_of_nodes())
		graph = max(nx.connected_component_subgraphs(G), key=len)
		print('LCC: # nodes', graph.number_of_nodes())
		query = query.UndirectedSingleLayer(graph)
		#oracle = oracle.Oracle(graph, dataset)
		log_file = log_file + dataset + '.txt'

	for i in range(0, int(args.experiment)):
		row = []

		tmp = []
		for type in exp_list:
			# sample = UndirectedSingleLayer(query, oracle, budget, \
			# 							   bfs_budget, type, dataset, log, k, is_cost,
			# 							   log_interval)
			sample = UndirectedSingleLayer(query, budget, bfs_budget, type, dataset, log, k, is_cost, log_interval)

			if starting_node == -1: starting_node = sample._query.randomNode()

			print('[{}] Experiment {} starts at node {}'.format(type, i, starting_node))



			# Getting sample
			sample.generate()
			# End getting sample

			cost_arr = Append_Log(sample, type)

			#Logging(sample)

		if 'budget' not in Log_result:
			Log_result['budget'] = cost_arr
		else:
			Log_result['budget'] += (cost_arr)

		starting_node = -1

	SaveToFile(Log_result)
	# Logging(sample)
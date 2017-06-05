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
from collections import defaultdict, Counter

from scipy import stats

import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt

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
		self._score_den_threshold = 0.1
		self._sample_graph = nx.Graph() # The sample graph
		self._nodes_observed_count = [] # Number of nodes observed in each iteration
		self._latest_edge_added = dict()
		self._flow_score = dict()
		self._avg_deg = 0.
		self._med_deg = 0.
		self._cost_spent = []
		self._nodes_return = []
		self._exp_count = 0
		self._densi_count = 0
		self._track_obs_nodes = []
		self._track_cost = []
		self._track = {}
		self._track_edges = {}
		self._track_cc = {}
		self._track_new_nodes = []
		self._track_selected_node = []
		self._track_rank_selected = []
		self._track_score_den = []

		self._track_hyb_samp = []
		self._track_hyb_samp_nn = []
		self._track_hyb_samp_threshold = []

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

		self._WINDOW_SIZE = 10
		self._PREV = 100
		self._SAMP = 1

		self._data_to_plot = []
		self._chose_deg = []

	def _expansion_random(self, candidate_list):
		degree_observed = self._sample_graph.degree(candidate_list)
		degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
		top_20 = int(.2 * len(degree_observed_sorted))
		sel = random.choice(degree_observed_sorted[top_20:])
		current_node = sel[0]

		return current_node

	def _expansion(self, candidate_list):
		count = 1
		DECAY_R = 0.005
		GROWTH_R = 0.01
		node_unobs_count = {}
		node_unobs = {}

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		#sub_sample['nodes']['open'].add(current_node)

		while self._cost < self._budget:
			####
			closed_nodes = list(self._sample['nodes']['close'])
			open_nodes = list(self._sample['nodes']['open'])

			degree_closed = self._sample_graph.degree(closed_nodes)
			degree_open = self._sample_graph.degree(open_nodes)
			degree_seq = sorted(degree_closed.values())
			####

			current_node = random.choice(open_nodes)

			if current_node not in sub_sample['nodes']['open']:
				sub_sample['nodes']['open'].add(current_node)


			# Make a query
			nodes, edges, c = self._query.neighbors(current_node)
			current_node_true_deg = len(nodes)
			self._count_new_nodes(nodes, current_node, deg_obs=degree_open[current_node])

			###
			percent_tile = stats.percentileofscore(degree_seq, current_node_true_deg)

			obs_nodes = self._sample_graph.nodes()
			new_nodes = set(nodes) - set(obs_nodes)
			node_unobs_count[current_node] = len(new_nodes)
			node_unobs[current_node] = new_nodes
			self._new_nodes.append(len(new_nodes))
			####

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			print('			Percentile {} {}'.format(percent_tile, self._percentile))
			if percent_tile >= self._percentile:
				self._percentile = self._percentile * math.pow((1 + GROWTH_R), count)
				break

			self._percentile = self._percentile*math.pow((1-DECAY_R), count)



		# Update the sample with the sub sample
		self._updateSample(sub_sample)
		obs_nodes = self._sample_graph.nodes()
		final_list = {}
		max_val = 0
		max_node = node_unobs.keys()[0]
		candidates = set()
		for n in node_unobs.keys():
			nbs = self._sample_graph.neighbors(n)
			open_n = set(nbs) - set(obs_nodes)
			open_count = len(open_n)

			if open_count >= max_val:
				max_node = n
				max_val = open_count
				candidates = list(open_n)

		if len(candidates) == 0:
			print(' - No candidate, pick open nodes instead ! -')
			candidates = list(self._sample['nodes']['open'])

		return random.choice(candidates)




		# print('Check .. closed: {}	open: {} cost: {}'.format(len(self._sample['nodes']['close']), len(self._sample['nodes']['open']), self._cost))
		#
		# # Pick new direction for densification
		# obs_nodes = self._sample_graph.nodes()
		# closed_nodes = set(self._sample['nodes']['close'])
		# closed_candidates = node_unobs_count.keys()
		# best_node, idx = _mylib.get_max_values_from_dict(node_unobs_count, closed_candidates)
		#
		# neighbors_unobs = node_unobs[best_node]
		#
		# # Pick one from the open nodes
		# candidates = neighbors_unobs - closed_nodes
		#
		# while len(candidates) == 0:
		# 	print('  -- no candidates from this node, pick from any open nodes')
		# 	print(closed_candidates)
		# 	closed_candidates.remove(best_node)
		#
		# 	print('Remove {}, remaining size {}'.format(best_node, len(closed_candidates)))
		#
		# 	if len(closed_candidates) == 0:
		# 		candidates = list(self._sample['nodes']['open'])
		# 		print('bbbb', len(candidates))
		# 		break
		# 	else:
		# 		best_node, idx = _mylib.get_max_values_from_dict(node_unobs_count, closed_candidates)
		# 		#sorted_x = sorted(node_unobs_count.items(), key=operator.itemgetter(1), reverse=True)
		# 		#best_node = sorted_x[0][0]
		#
		# 		neighbors_unobs = node_unobs[best_node]
		# 		candidates = neighbors_unobs - closed_nodes
		#
		# 		print('Best node', best_node)
		#
		# #print(' -- # best unobs {} - candidate {} / {}'.format(best_node_unobs, len(candidates), len(neighbors_unobs) ))
		# print('before return ', len(candidates))
		# return random.choice(list(candidates))



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

	def _expansion_oracle(self, candidate_list):
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
		#sub_sample['nodes']['open'].update(self._sample['nodes']['open'])
		#sub_sample['nodes']['close'].update(self._sample['nodes']['close'])
		#sub_sample['edges'].update(self._sample['edges'])


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
				# degree_observed_sorted = _mylib.sortDictByValues(degree_observed,reverse=True)
				# current = degree_observed_sorted[0][0]
				# current_node_obs_deg = degree_observed_sorted[0][1]

				# degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
				# top_20 = int(.2 * len(degree_observed_sorted))
				#
				# if top_20 != 0:
				# 	sel = random.choice(degree_observed_sorted[:top_20])
				# 	# current = sel[0]
				# 	# current_node_obs_deg = sel[1]
				# 	current = degree_observed_sorted[0][0]
				# 	current_node_obs_deg = degree_observed_sorted[0][1]
				# else:
				# 	current =degree_observed_sorted[0][0]
				# 	current_node_obs_deg =degree_observed_sorted[0][1]
				current, current_node_obs_deg = self._node_selection(den_nodes)
			else:
				current = list(den_nodes)[0]
				current_node_obs_deg = degree_observed[current]

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current)
			self._count_new_nodes(nodes, current, deg_obs=current_node_obs_deg, score=score_den)

			close_nodes = set(sub_sample['nodes']['close']).union(set(self._sample['nodes']['close']))
			not_close_nbs = set(nodes) - close_nodes

			# Update edges time added
			for node in nodes:
				self._latest_edge_added[node] = self._cost
				if node not in close_nodes:
					self._flow_score[node] = self._flow_score.get(node, 1) + (self._flow_score.get(current, 1)/len(not_close_nbs))

			self._latest_edge_added.pop(current, None)
			self._flow_score.pop(current, None)
			# Update edges time added

			# TODO: Densification score, to be changed.
			# Update the densification and expansion scores
			close_n = set(self._sample['nodes']['close']) | set(sub_sample['nodes']['close'])
			deg_close = self._sample_graph.degree(list(close_n)).values()
			avg_deg = np.average(np.array(deg_close))
			sd_deg = np.std(np.array(deg_close))


			score_den = self._scoreDen_test(sub_sample, avg_deg, nodes,current_node_obs_deg,score_den)

			# Store the densification and expansion scores
			self._score_den_list.append(score_den)

			self._densi_count += 1

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			# TODO: just a dummy statement for MOD method
			#T = self._score_den_threshold * math.exp(-(0.05 * self._exp_count))
			T = 0.2
			# c_nodes = set(sub_sample['nodes']['close'])
			# deg = self._sample_graph.degree(list(c_nodes))
			# mean_deg = np.average(np.array(deg.values()))
			# sd_deg = np.std(np.array(deg.values()))
			# T = mean_deg + sd_deg


			#print('T {} \t score_den: {}		cost: {} {} {} '.format(round(T,2), round(score_den,2), self._cost, round(mean_deg,2), round(sd_deg,2)))

			if score_den <= T:
			#if score_den < (-1*sd_deg):
				break


		#self._score_den_threshold = T

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

			current_node = self._cal_score(candidates)

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
			self._count_new_nodes(nodes, current)


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

	def _scoreDen_test(self, sub_sample, avg_deg=0, nodes=None, deg_obs=0, prev=0):
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
		# new_nodes = nodes.difference(sub_sample['nodes']['close']) \
		# 	.difference(sub_sample['nodes']['open']) \
		# 	.difference(self._sample['nodes']['open']).difference(self._sample['nodes']['close'])
		observed_nodes = self._sample_graph.nodes()
		new_nodes = set(nodes) - set(observed_nodes)
		deg_true = len(nodes)
		deg_new_nodes = len(new_nodes)
		deg_existing_open = len(nodes) - deg_new_nodes - deg_obs
		deg_in = deg_existing_open + deg_new_nodes
		try:
			ratio = (deg_new_nodes / deg_in) * (deg_true / deg_obs)
		except ZeroDivisionError:
			ratio = 0.

		# TODO: score function
		# Calculate the densification score
		try:
			#score = deg_true - avg_deg
			score = deg_new_nodes / deg_true #deg_in
			#score = deg_true
		except ZeroDivisionError:
			score = 0.

		# Store number of new nodes for logging later
		self._new_nodes.append(deg_new_nodes)

		if np.isfinite(prev):
			s = (0.5 * prev) + (self._wt_den * score)
		else:
			s = self._wt_den * score
		#s = score

		return s

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
		except KeyError as e:
			print('		Error update Subsample:', e, candidate)
			nodes = self._query._graph.nodes()
			print(candidate in set(nodes))


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
			self._count_new_nodes(nodes, current)

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

			self._count_new_nodes(nodes, current)

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

		rank=0
		deg_observed=0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			# For tracking
			self._count_new_nodes(nodes, current_node, rank,deg_observed)

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
				print(' Walking.. {} neighbors'.format(len(nodes)))

			current_node = random.choice(candidates)

			open_nodes = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(
					self._sample['nodes']['close']))
			deg_observed = self._sample_graph.degree(current_node)
			#rank = self._get_node_rank_from_excess_degree(current_node, open_nodes)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _count_new_nodes(self, nodes, current, rank=0, deg_obs=0, score=0):
		current_nodes = self._sample_graph.nodes()
		new_nodes = set(nodes).difference(current_nodes)

		deg_true = len(nodes)
		deg_new_nodes = len(new_nodes)
		deg_existing_open = len(nodes) - deg_new_nodes - deg_obs
		deg_in = deg_existing_open + deg_new_nodes
		try:
			ratio = (deg_new_nodes / deg_in) * (deg_true / deg_obs)
		except ZeroDivisionError:
			ratio = 0.


		self._track_selected_node.append(current)
		self._track_rank_selected.append(deg_obs)
		self._track_new_nodes.append(score)
		self._track_score_den.append(score)

	def _bandit(self):
		# Initialize graph with Random Walk
		close_n, cash_count = self._init_bandit_g_rw()

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}

		# Start Bandit algorithm
		arms, node2arm = self._get_arms(close_n)

		bandit = {'arms': arms,
				  'score': dict.fromkeys(arms.keys(), float('inf')),
				  'count': dict.fromkeys(arms.keys(), 0),
				  'created_at': dict.fromkeys(arms.keys(), 0),
				  'rewards': defaultdict(list),
				  'node2arm': node2arm}

		# Initialize score for each arm
		for k,v in bandit['arms'].iteritems():
			count = len(k.split('.')) - 1
			bandit['score'][k] = count
			#bandit['score'][k] = (1 / count) * len(v)


		# Pick fist arm
		max_score = max(bandit['score'].values())
		candidate_arms = _mylib.get_keys_by_value(bandit['score'], max_score)
		current_arm = random.choice(candidate_arms)
		members = bandit['arms'][current_arm]
		current_node = random.choice(members)

		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])
		sub_sample['nodes']['close'].update(close_n)
		iter_count = 1

		#cash_count = {}


		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query on the selected node
			nodes, edges, c = self._query.neighbors(current_node)
			new_nodes = set(nodes) - set(self._sample_graph.nodes())
			closed_nodes = self._sample['nodes']['close'] | sub_sample['nodes']['close']

			# Update bandit
			bandit = self._update_arms(bandit, current_node, current_arm, nodes, closed_nodes, iter_count)

			for node in nodes:
				cash_count[node] = cash_count.get(node, 1 ) + (cash_count.get(current_node,1) / len(nodes))


			# For tracking
			self._count_new_nodes(nodes, current_node)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the cost
			self._increment_cost(c)

			bandit = self._update_score(bandit, iter_count)
			max_score = max(bandit['score'].values())

			candidate_arms = _mylib.get_keys_by_value(bandit['score'], max_score)
			current_arm = random.choice(candidate_arms)
			members = bandit['arms'][current_arm]
			#current_node = self._pick_next_node(members)#random.choice(members)
			current_node = self._pick_next_node_from_cash(members, cash_count)

			iter_count += 1

		self._updateSample(sub_sample)

	def _pick_next_node(self, candidates):
		return random.choice(candidates)

	def _pick_next_node_from_cash(self, candidates, cash_count):
		cash_count_sorted = _mylib.sortDictByValues(cash_count, reverse=True)

		for index, ccc in enumerate(cash_count_sorted):
			#print(cash_count_sorted[index][0])
			if cash_count_sorted[index][0] in candidates:
				c = cash_count_sorted[index][0]
				return c




	def _pr_graph_tool(self):
		gt_graph = networkx2gt.nx2gt(self._sample_graph)
		pr = graph_tool.centrality.pagerank(gt_graph)

		max_node = candidates[0]
		max_val = 0
		for cand in candidates:
			val = pr[gt_graph.vertex(int(cand))]
			if val > max_val:
				max_node = cand

		return max_node



	def _get_arms_members_more_than(self, bandit, k=10):
		candidates = []
		for key, members in bandit['arms'].iteritems():
			mem_count = len(members)
			if mem_count > k:
				candidates.append(key)
		return candidates

	def _update_score(self, bandit, iter_count, CONSTANT_P = 150):


		next_iter = iter_count + 1
		CONSTANT_P = CONSTANT_P * math.exp(-(0.05 * next_iter))
		#print(CONSTANT_P)

		for arm_id, rewards in bandit['rewards'].iteritems():
			avg_rewards = np.average(np.array(rewards))
			#avg_rewards = np.average(np.array(rewards),weights=range(1, len(rewards) + 1))


			created_at = bandit['created_at'][arm_id]
			arm_picked_count = bandit['count'][arm_id] + 0.1

			score = avg_rewards + \
					(CONSTANT_P * math.sqrt( (2 * math.log(next_iter - created_at)) / arm_picked_count ))

			#print(score, arm_picked_count, avg_rewards, rewards)
			bandit['score'][arm_id] = score

		return bandit

	def _update_arms(self, bandit, current_node, current_arm, nbs, closed_nodes, iter_count):
		#print('Update arms', iter_count)
		C = 5
		open_nodes = set(self._sample_graph.nodes()) - closed_nodes

		# all nodes in current arm
		members_in_arms = bandit['arms'][current_arm]
		# nbs that are in the current arm
		nbs_cur_arm = set(nbs).intersection(set(members_in_arms))
		# nbs that are newly added nodes
		new_nodes = set(nbs) - set(self._sample_graph.nodes())
		# nbs that are not in the current arm
		nbs_out_arm = (set(nbs) - nbs_cur_arm) & open_nodes

		# nbs that are existing inside and outside current arm
		nbs_in_out_arm = nbs_cur_arm & nbs_out_arm

		#print('Nbs: {}, Current arm: {}/{}, Outside: {}, Newly added: {}'.format(len(nbs), len(nbs_cur_arm), len(members_in_arms), len(nbs_out_arm), len(new_nodes)))

		# Update info of the current arm
		# Remove current node from current arm
		bandit['arms'][current_arm].remove(current_node)
		bandit['node2arm'].pop(current_node, -1)
		bandit['rewards'][current_arm].append(len(new_nodes))
		bandit['count'][current_arm] += 1

		avg_cur_arm_reward = np.mean(np.array(bandit['rewards'][current_arm]))
		P = (len(new_nodes)+1) / len(nbs)
		reward_4_new_arm = (P * avg_cur_arm_reward)

		#print ('	Avg. Reward: {}, Score: {}, R4new: {}, P: {}'.format(avg_cur_arm_reward, bandit['score'][current_arm], reward_4_new_arm, P))

		# For the case that current arm has one node and it is selected.
		if len(bandit['arms'][current_arm]) == 0:
			bandit['arms'].pop(current_arm)
			bandit['score'].pop(current_arm)
			bandit['count'].pop(current_arm)
			bandit['created_at'].pop(current_arm)
			bandit['rewards'].pop(current_arm)

		# # CASE 1: New arm becuase current node has link with nodes in current arm
		# for index, node in enumerate(nbs_cur_arm):
		# 	# New arm id
		# 	new_arm = current_arm + str(current_node) + '.'
		# 	# Update arm id
		# 	bandit['node2arm'][node] = new_arm
		# 	# Add node to new arm
		# 	bandit['arms'][new_arm].append(node)
		# 	#bandit['score'][new_arm] = bandit['score'].get(new_arm, float('inf'))
		# 	bandit['count'][new_arm] = 0
		# 	bandit['created_at'][new_arm] = iter_count
		# 	if index == 0:
		# 		count = len(new_arm.split('.'))
		# 		reward = reward_4_new_arm + ((1 - P) * (1/(count-1) ))
		# 		bandit['rewards'].setdefault(new_arm, []).append(reward)
		# 		bandit['score'][new_arm] = bandit['score'].get(new_arm, reward)
		#
		# 		#avg_cur_arm_reward = np.mean(np.array(bandit['rewards'][new_arm]))
		# 		#bandit['score'][current_arm] = avg_cur_arm_reward
		#
		# 	# Remove node from current arm
		# 	bandit['arms'][current_arm].remove(node)
		#
		# 	# If current Arm is disappear
		# 	if len(bandit['arms'][current_arm]) == 0:
		# 		bandit['arms'].pop(current_arm)
		# 		bandit['score'].pop(current_arm)
		# 		bandit['count'].pop(current_arm)
		# 		bandit['created_at'].pop(current_arm)
		# 		bandit['rewards'].pop(current_arm)

		# CASE 2: New arm becuase of newly added nodes
		for index, node in enumerate(new_nodes):
			# New arm id
			new_arm = str(current_node) + '.'
			# Update arm id
			bandit['node2arm'][node] = new_arm
			# Update arm info
			bandit['arms'][new_arm].append(node)

			if index == len(new_nodes) - 1:
				close_count = len(new_arm.split('.')) - 1
				reward = reward_4_new_arm + ((1 - P) * ((close_count)))
				#reward = reward_4_new_arm + ((1 - P) * (len(new_nodes) / (close_count)))

				bandit['rewards'].setdefault(new_arm, []).append(reward)
				bandit['created_at'][new_arm] = bandit['created_at'].get(new_arm, iter_count)
				bandit['count'][new_arm] = bandit['count'].get(new_arm, 0)


		# CASE 3: New arm becuase current node connects with nodes outside.
		for index, node in enumerate(nbs_in_out_arm):
			# Current arm of particular node
			node_arm = bandit['node2arm'][node]
			# New arm id
			new_arm = node_arm + str(current_node) + '.'
			# Update arm id
			bandit['node2arm'][node] = new_arm
			# Add node to new arm
			bandit['arms'][new_arm].append(node)

			bandit['count'][new_arm] = bandit['count'].get(new_arm, 0)
			bandit['created_at'][new_arm] = bandit['created_at'].get(new_arm, iter_count)

			#bandit['rewards'].setdefault(new_arm, []).append(reward)

			close_count = len(new_arm.split('.')) - 1
			node_in_arms = len(bandit['arms'][new_arm])
			reward = reward_4_new_arm + ((1 - P) * (close_count))
			#reward = reward_4_new_arm + ((1 - P) * (node_in_arms/close_count) )
			bandit['rewards'][new_arm] = [reward]

			# Remove node from current arm
			bandit['arms'][node_arm].remove(node)

			# If current Arm is disappear
			if len(bandit['arms'][node_arm]) == 0:
				bandit['arms'].pop(node_arm)
				bandit['score'].pop(node_arm, -1)
				bandit['count'].pop(node_arm)
				bandit['created_at'].pop(node_arm)
				bandit['rewards'].pop(node_arm, -1)

		return bandit

	def _get_arms(self, close_n):
		sample_g = self._sample_graph
		open_nodes = set(sample_g.nodes()) - close_n

		arms = defaultdict(list)
		node2arm = dict()

		for o_n in open_nodes:
			nbs = sample_g.neighbors(o_n)
			key = ''
			for x in nbs:
				key += x +'.'

			arms[key].append(o_n)
			node2arm[o_n] = key

		print(len(node2arm), len(open_nodes))
		return arms, node2arm

	def _init_bandit_g_rw(self, START_BUDGET = 10):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)

		cash_count = {}

		while self._cost < START_BUDGET and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			# For tracking
			self._count_new_nodes(nodes, current_node)

			for node in nodes:
				cash_count[node] = cash_count.get(node, 1 ) + (cash_count.get(current_node,1) / len(nodes))

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
				current_node = random.choice(list(nodes))
				# Query the neighbors of current
				nodes, edges, c = self._query.neighbors(current_node)
				# Candidate nodes are the (open) neighbors of current node
				candidates = list(
					set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
				print(' Walking.. {} neighbors'.format(len(nodes)))

			current_node = random.choice(candidates)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)
		close_n = set(sub_sample['nodes']['close']) | set(self._sample['nodes']['close'])
		return close_n, cash_count

	def _cash_spread(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		cash_count = {}

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node)

			for c_n in nodes:
				cash_count[c_n] = cash_count.get(c_n, 1) + (1 / len(nodes))

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

			#degree_observed = self._sample_graph.degree(candidates)
			cash_count_sorted = _mylib.sortDictByValues(cash_count, reverse=True)


			for index,ccc in enumerate(cash_count_sorted):
				if cash_count_sorted[index][0] in candidates:
					current_node = cash_count_sorted[index][0]
					break



		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_obs_deg(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		rank = 0
		deg_obs = 0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node, rank, deg_obs)

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
			degree_true = self._query._graph.degree(candidates)
			degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
			degree_true_sorted =  _mylib.sortDictByValues(degree_true, reverse=True)

			# top_20 = int(.2 * len(degree_observed_sorted))
			# sel = random.choice(degree_observed_sorted[:top_20])
			# current_node = sel[0]
			# deg_obs = sel[1]

			current_node = degree_observed_sorted[0][0]
			deg_obs = degree_observed_sorted[0][1]



			# pr = nx.clustering(self._sample_graph)
			# cores = nx.core_number(self._sample_graph)
			# nbs_deg = nx.average_neighbor_degree(self._sample_graph, candidates)
			#
			#
			# #b = [x[1] for x in degree_true_sorted[:10]]
			# aa = ([x[1] for x in degree_observed_sorted[:15]])
			# a = ([degree_true[x[0]] for x in degree_observed_sorted[:15]])
			# b = ([round(pr[x[0]], 2) for x in degree_observed_sorted[:15]])
			# dd = [nbs_deg[x[0]] for x in degree_observed_sorted[:15] ]
			#
			# score = self._neighbor_score(candidates, set(sub_sample['nodes']['close']))
			#
			# ee = ([round(score[x[0]], 2) for x in degree_observed_sorted[:15]])



			# print(deg_obs, degree_true[current_node], '---', max(a))
			# print('		obs', aa)
			# print('		tru', a)
			# print('		ccc', b)
			# print('		nbs', dd)
			# print('		pro', ee)








			#rank = nx.average_clustering(self._sample_graph, list(sub_sample['nodes']['close']))
			#print(self._cost, rank)
			#rank = self._get_node_rank_from_excess_degree(current_node, candidates)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_avg_nbs(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		rank = 0
		deg_obs = 0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node, rank, deg_obs)

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = list(
				set(self._sample_graph.nodes()).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))

			# Node Selection
			current_node, deg_obs = self._node_selection(candidates)

			# current_node = degree_observed_sorted[0][0]
			# deg_obs = degree_observed_sorted[0][1]





		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _node_selection(self, candidates):
		graph = self._sample_graph
		# Select by average nbs degree
		# score_d = nx.average_neighbor_degree(self._sample_graph, nodes=candidates)

		obs_deg = graph.degree(candidates)
		obs_deg_new = {k: v for k, v in obs_deg.items() if v > 2}
		#deg_all_new = {k: v for k, v in deg_all.items() if v > 2}

		# if len(obs_deg_new) < 2:
		# 	selected_node = random.choice(obs_deg.keys())
		# 	deg_node = obs_deg[selected_node]
		# 	return selected_node, deg_node

		sorted_list = _mylib.sortDictByValues(obs_deg, reverse=True)
		top_k = int(.2)*len(sorted_list)

		if top_k == 0:
			top_k = len(sorted_list)
		if top_k > 50:
			top_k = 50

		min_val = min(obs_deg.values())
		max_val = max(obs_deg.values())
		score_d = dict()

		min_flow = min(self._flow_score.values())
		max_flow = max(self._flow_score.values())

		top_k_nodes = []
		p_members = dict()
		for item in sorted_list[:top_k]:
			k = item[0]
			v = item[1]
			cc = nx.clustering(graph, k)
			flow_s = self._flow_score[k]
			# node_p_label = p[k]
			# com_size = 0
			# if node_p_label in p_members.keys():
			# 	com_size = p_members[node_p_label]
			# else:
			# 	com_size = len(_mylib.get_members_from_com(node_p_label, p))

			try:
				score_d[k] = ((v - min_val) / (max_val - min_val)) * (1 - cc)
				#score_d[k] *= com_size / _sub_g.number_of_nodes()
				#score_d[k] *= ((avg_nbs_deg[k] - min_val) / (max_val - min_val))
				#score_d[k] = (0.5*a) + (0.5*b)
			except ZeroDivisionError:
				score_d[k] = (v ) * (1 - cc)

			top_k_nodes.append(self._query._graph.degree(k))


		sorted_list = _mylib.sortDictByValues(score_d, reverse=True)
		selected_node = sorted_list[0][0]
		deg_node = sorted_list[0][1]
		sel_deg = self._query._graph.degree(selected_node)

		####
		deg = self._query._graph.degree(candidates)
		avg_nbs_deg = nx.average_neighbor_degree(self._sample_graph, nodes=candidates)
		true_deg = list()
		score = list()
		obs_deg = list()
		avg_nbs = list()
		for t in sorted_list:
			k = t[0]
			true_deg.append(deg[k])
			score.append(round(t[1],2))
			obs_deg.append(self._sample_graph.degree(k))
			avg_nbs.append(int(avg_nbs_deg[k]))

		if self._cost % 10 == 0:
			if len(self._chose_deg) == 0:
				self._chose_deg.append(0)
			self._data_to_plot.append(true_deg)
			self._chose_deg.append(true_deg[0])

		# print('T -', max(true_deg), int(np.average(np.array(true_deg))), ':', true_deg)
		# print('O -', max(obs_deg), int(np.average(np.array(obs_deg))), ':',  obs_deg)
		# print(score)

		# tau, p_value = stats.kendalltau(true_deg, avg_nbs)
		# print(tau, round(p_value,2))
		# self._line_1.append(tau)


		# max_deg = max(deg.values())
		# avg_deg = np.average(np.array(deg.values()))
		# indices = np.where(np.array(deg.values()) == max_deg)[0]
		# best = np.array(deg.keys())[indices][0]
		# deg_obs = self._sample_graph.degree(candidates)
		# max_deg_obs = max(deg_obs.values())
		#
		#
		# print('\t Max:{}, Avg:{} | deg: {} s:{} '.format(max_deg, int(avg_deg), sel_deg, round(deg_node,2)))
		# print('\t', true_deg[:15])
		# print('\t', indices, best, deg_obs[best], max_deg_obs)

		####

		return selected_node, deg_node

	def _max_score(self):
		candidates = self._sample['nodes']['open']

		current_node = starting_node

		# End

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

			current_node = self._pick_from_close(candidates)
			#current_node = self._cal_score(candidates)
			#score_sorted = _mylib.sortDictByValues(score, reverse=True)

			#current_node = score_sorted[0][0]

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _pick_from_close(self, candidates):
		close_nodes = set(self._sample_graph.nodes()) - set(candidates)
		deg = self._sample_graph.degree(close_nodes)
		print(' pick from ', len(deg), len(close_nodes))

		if len(deg) <= 10:
			return random.choice(list(candidates))

		sorted_deg = _mylib.sortDictByValues(deg,reverse=True)
		new_cand = set()

		for t in sorted_deg[:10]:
			n = t[0]
			nbs = self._sample_graph.neighbors(n)
			open_nb = set(nbs).intersection(candidates)
			print(len(open_nb))
			new_cand.update(open_nb)

		if len(new_cand) == 0:
			new_cand = candidates

		return random.choice(list(new_cand))

	def _cal_score(self, candidates):
		open_nodes = candidates
		close_nodes = set(self._sample_graph.nodes()) - set(candidates)

		degree_cand = self._sample_graph.degree(candidates)
		degree_close = self._sample_graph.degree(close_nodes)
		degree_avg_close = np.mean(np.array(degree_close.values()))

		pos_list = {}
		neg_list = {}
		for candidate in candidates:
			deg = degree_cand[candidate]
			deg_diff = deg - degree_avg_close
			print('			deg',deg)
			if deg_diff >= 0 :
				pos_list[candidate] = deg_diff
			else:
				neg_list[candidate] = deg_diff

		print('Total: {} -- Pos{} Neg{}'.format(len(candidates), len(pos_list),len(neg_list)))

		if len(pos_list) != 0:
			print('		[Cal S] Positive list', degree_avg_close)
			n = _mylib.pickMaxValueFromDict(pos_list)
		else:
			print('		[Cal S] Negative list', degree_avg_close)
			n = _mylib.pickMaxValueFromDict(neg_list)
			#n = random.choice(list(neg_list.keys()))


		return n

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

	def _neighbor_score(self, candidates, close_nodes):
		score = dict()
		cand_score = dict()
		for n in candidates:
			nbs = self._sample_graph.neighbors(n)
			s = []
			for nb in nbs:
				if nb not in score.keys():
					nbs_of_nb = self._sample_graph.neighbors(nb)
					nbs_close = set(nbs_of_nb).intersection(close_nodes)
					score[nb] = len(nbs_close) / len(nbs_of_nb)

				s.append(score[nb])
			cand_score[n] = np.mean(np.array(s))

		return cand_score

	def _hybrid(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		is_MOD = True
		new_nodes_step = []

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			# For tracking
			self._count_new_nodes(nodes, current_node)

			# New nodes found
			new_nodes = set(nodes).difference(set(self._sample_graph.nodes()))
			new_nodes_step.append(len(new_nodes))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost
			self._increment_cost(c)

			if len(new_nodes_step) % self._WINDOW_SIZE == 0:
				is_MOD = self._decision_making(new_nodes_step)

			# TODO: Need some criteria to switch between MOD and RW !
			if is_MOD:
				#print(' -- MOD {} neighbors {} new nodes'.format(len(nodes), len(new_nodes)))
				closed_nodes = set(sub_sample['nodes']['close']).union(set(self._sample['nodes']['close']))
				open_nodes = set(self._sample_graph.nodes()).difference(closed_nodes)

				deg = self._sample_graph.degree(open_nodes)
				deg_sorted =  _mylib.sortDictByValues(deg,reverse=True)

				current_node = deg_sorted[0][0]
			else:
				if len(new_nodes_step) % 10 == 0:
					if self._track_hyb_samp[-2] != -1:
						closed_nodes = set(sub_sample['nodes']['close']).union(set(self._sample['nodes']['close']))
						current_node = random.choice(list(closed_nodes))
						# Query the neighbors of current
						nodes, edges, c = self._query.neighbors(current_node)
						# Candidate nodes are the (open) neighbors of current node
						new_nodes = list(
							set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))


				candidates = new_nodes
				while len(candidates) == 0:
					current_node = random.choice(list(nodes))
					# Query the neighbors of current
					nodes, edges, c = self._query.neighbors(current_node)
					# Candidate nodes are the (open) neighbors of current node
					candidates = list(
						set(nodes).difference(sub_sample['nodes']['close']).difference(self._sample['nodes']['close']))
					#print(' ---- Walking.. {} neighbors'.format(len(nodes)))

				current_node = random.choice(list(candidates))

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _decision_making(self, steps):
		D = 0.15
		WINDOW_SIZE = self._WINDOW_SIZE
		curr_window = np.array(steps[-WINDOW_SIZE:])
		prev_window = np.array(steps[-(2*WINDOW_SIZE):-WINDOW_SIZE])
		t = len(steps)

		curr_window_sum = np.sum(curr_window)
		prev_window_sum = np.sum(prev_window)

		self._track_hyb_samp_nn.append(curr_window_sum)
		self._track_hyb_samp.append(self._SAMP)

		if t == self._WINDOW_SIZE:
			self._PREV = (curr_window_sum)
			self._track_hyb_samp_threshold.append(0)
			return True

		#THRESHOLD = math.log((self._PREV * math.pow((1 - D), (t / WINDOW_SIZE))))
		exponent = (t/WINDOW_SIZE)
		THRESHOLD = int(self._PREV * math.exp(-exponent/200))

		self._track_hyb_samp_threshold.append(THRESHOLD)
		self._PREV = np.mean(np.array(self._track_hyb_samp_nn[:-1] ))#THRESHOLD #curr_window_sum

		if curr_window_sum < THRESHOLD:
			self._SAMP = self._SAMP * -1


		if self._SAMP == -1:
			print(' RW! sum {} .. threshold {}'.format(curr_window_sum, THRESHOLD))
			return False
		elif self._SAMP == 1:
			print(' MOD! sum {} .. threshold {}'.format(curr_window_sum, THRESHOLD))
			return True

	def _get_max_open_nbs_node(self, close_n):

		count_n = {}
		for n in close_n:
			nodes = self._sample_graph.neighbors(n)
			candidates = list(set(nodes).difference(close_n))
			count_n[n] = len(candidates)

		max_count = max(count_n.values())
		print('Max count', max_count)

		candidates = _mylib.get_members_from_com(max_count, count_n)
		return random.choice(candidates)

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
			obs_edges = self._sample_graph.number_of_edges()

			c = int(self._cost)

			self._track[c] = obs_nodes
			self._track_edges[c] = obs_edges

			self._track_k.append(self._tmp)
			self._track_open.append(len(self._sample['nodes']['open']))


			nodes_count = self._sample_graph.number_of_nodes()
			edges_count = self._sample_graph.number_of_edges()

			#self._line_1.append(nodes_count)
			#self._line_2.append(edges_count)

	def _max_excess_deg(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])
		rank = 0
		deg_observed = 0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node, rank=rank, deg_obs=deg_observed)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = set(sub_sample['nodes']['open'])

			degree_excess = self._get_node_excess_degree(candidates)

			degree_excess_sorted = _mylib.sortDictByValues(degree_excess, reverse=True)
			current_node = degree_excess_sorted[0][0]
			#rank = self._get_node_rank_from_excess_degree(current_node, candidates)
			deg_observed = self._sample_graph.degree(current_node)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)


	def _k_rank(self):
		current_node = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current_node)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		rank=0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c = self._query.neighbors(current_node)
			self._count_new_nodes(nodes, current_node, rank)


			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current_node)

			# Add edges to sub_graph
			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])
			# Update the cost

			self._increment_cost(c)

			candidates = set(sub_sample['nodes']['open'])

			degree_true = self._query._graph.degree(candidates)
			degree_observed = self._sample_graph.degree(candidates)

			degree_excess = Counter(degree_true)
			degree_excess.subtract(Counter(degree_observed))

			degree_excess_sorted = _mylib.sortDictByValues(degree_excess, reverse=True)

			cutoff = int(len(degree_excess_sorted) * 0.1)
			if cutoff > 2:
				current_node = random.choice(degree_excess_sorted[:cutoff])[0]
			else:
				current_node = degree_excess_sorted[0][0]

			rank = self._get_node_rank_from_excess_degree(current_node, candidates)



		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _get_node_excess_degree(self, candidates):
		degree_true = self._query._graph.degree(candidates)
		degree_observed = self._sample_graph.degree(candidates)

		degree_excess = Counter(degree_true)
		degree_excess.subtract(Counter(degree_observed))

		return degree_excess

	def _get_node_rank_from_excess_degree(self, selected_node, candidates):
		degree_excess = self._get_node_excess_degree(candidates)
		degree_excess_val = np.array(degree_excess.values())
		degree_excess_val[::-1].sort()

		selected_ex_deg = degree_excess[selected_node]

		ranks = np.where(degree_excess_val == selected_ex_deg)[0]
		avg_rank = np.mean(ranks)

		min = degree_excess_val.min()
		max = degree_excess_val.max()
		mean = np.mean(degree_excess_val)


		print('Min {}, Max {}, Avg. {}'.format(min, max, mean))

		return avg_rank / len(candidates)








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
				if len(current_list) != 0:
					self._stage = 'exp'
					if self._exp_type == 'oracle':
						current_list = self._getExpNodes()
						current = self._expansion_oracle(current_list)
						self._exp_count += 1
					elif self._exp_type == 'random-exp':
						current = self._expansion_random(self._sample['nodes']['open'])
						self._exp_count += 1
					elif self._exp_type == 'exp-den':
						print('		Start Expansion ..')
						#current = self._expansion(current_list)
						current = self._expansion_random(self._sample['nodes']['open'])
						self._exp_count += 1
				else:
					current = starting_node

				# Perform densification
				self._stage = 'den'
				if self._exp_type == 'random':
					self.random_sampling()
				elif self._exp_type == 'rw':
					self._random_walk()
				elif self._exp_type == 'mod':
					self._max_obs_deg()
				elif self._exp_type == 'med':
					self._max_excess_deg()
				elif self._exp_type == 'max-score':
					self._max_score()
				elif self._exp_type == 'oracle':
					current_list = self._densification_oracle(current)
				elif self._exp_type == 'sb':
					self._snowball_sampling()
				elif self._exp_type == 'bfs':
					self._bfs()
				elif self._exp_type == 'hybrid':
					self._hybrid()
				elif self._exp_type == 'vmab':
					self._bandit()
				elif self._exp_type == 'cash':
					self._cash_spread()
				elif self._exp_type == 'k-rank':
					self._k_rank()
				elif self._exp_type == 'exp-den':
					current_list = self._densification(current)
				elif self._exp_type == 'man':
					self._max_avg_nbs()


				self._densi_count += 1

				#print('			Budget spent: {}/{}'.format(self._cost, self._budget))

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

def SaveToFile(results_nodes,results_edges, query_order, rank_order):
	log.save_to_file(log_file_node, results_nodes)
	#log.save_to_file(log_file_edge, results_edges)
	#log.save_to_file(log_file_order, query_order)
	log.save_to_file(log_file_rank, rank_order)


def Append_Log(sample, type):
	track_sort = _mylib.sortDictByKeys(sample._track)
	track_edges_sort = _mylib.sortDictByKeys(sample._track_edges)
	cost_track = [x[0] for x in track_sort]
	obs_track = [x[1] for x in track_sort]
	obs_edges_track = [x[1] for x in track_edges_sort]

	Log_result[type] = Log_result.get(type, list()) + obs_track
	Log_result_edges[type] = Log_result_edges.get(type, list()) + obs_edges_track
	Log_result_step_sel_node[type] = Log_result_step_sel_node.get(type, list()) + sample._track_selected_node
	Log_result_step_sel_rank[type] = Log_result_step_sel_rank.get(type, list()) + sample._track_rank_selected
	Log_result_step_sel_rank[type+'.new'] = Log_result_step_sel_rank.get(type+'.new', list()) + sample._track_new_nodes


	return cost_track

def Append_Log_Hybrid(sample, trial):
	a = len(sample._track_hyb_samp)
	trial_l = [trial] * int(a)
	windows = [x for x in xrange(1,int(a)+1)]

	#if len(Log_result_hyb) == 0:
	Log_result_hyb['mode'] = Log_result_hyb.get('mode',[]) + sample._track_hyb_samp
	Log_result_hyb['new_nodes'] = Log_result_hyb.get('new_nodes', []) + sample._track_hyb_samp_nn
	Log_result_hyb['trial'] = Log_result_hyb.get('trial', []) + trial_l
	Log_result_hyb['window'] = Log_result_hyb.get('window', []) + windows
	Log_result_hyb['threshold'] = Log_result_hyb.get('threshold', []) + sample._track_hyb_samp_threshold

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-task', help='Type of sampling', default='undirected_single')
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-budget', help='Total budget', type=int, default=0)
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


	if mode == 1:
		#exp_list = ['med','mod','rw','exp-den']
		exp_list = ['med','mod','rw','exp-den']
	elif mode == 2:
		exp_list = ['mod','rw','exp-den']
	elif mode == 3:
		exp_list = ['exp-den']


	print(exp_list)
	Log_result = {}
	Log_result_edges = {}
	Log_result_step_sel_node = {}
	Log_result_step_sel_rank = {}

	Log_result_hyb = {}

	if dataset == None:
		f = fname.split('.')[1].split('/')[-1]
		dataset = f

	if args.task == 'undirected_single':
		G = _mylib.read_file(fname)

		print('Original: # nodes', G.number_of_nodes())
		graph = max(nx.connected_component_subgraphs(G), key=len)
		print('LCC: # nodes', graph.number_of_nodes())
		query = query.UndirectedSingleLayer(graph)
		#oracle = oracle.Oracle(graph, dataset)

		log_file_node = log_file + dataset + '_n.txt'
		log_file_edge = log_file + dataset + '_e.txt'
		log_file_order = log_file + dataset + '_order.txt'
		log_file_rank = log_file + dataset + '_rank.txt'

		n = graph.number_of_nodes()

		if budget == 0:
			budget = int(.10*n)
		print('{} Budget set to {} , n={}'.format(dataset, budget, n))

	print('-'*10)
	print(nx.info(graph))
	print('-' * 10)


	for i in range(0, int(args.experiment)):
		row = []

		tmp = []
		for type in exp_list:
			# sample = UndirectedSingleLayer(query, oracle, budget, \
			# 							   bfs_budget, type, dataset, log, k, is_cost,
			# 							   log_interval)
			sample = UndirectedSingleLayer(query, budget, bfs_budget, type, dataset, log, k, is_cost, log_interval)

			if starting_node == -1:
				#starting_node = sample._query.randomHighDegreeNode()
				starting_node = sample._query.randomFromLargeCommunity(graph, dataset)

			print('[{}] Experiment {} starts at node {}'.format(type, i, starting_node))


			# Getting sample
			sample.generate()
			# End getting sample

			cost_arr = Append_Log(sample, type)

			if type == 'hybrid':
				Append_Log_Hybrid(sample, i)

		Log_result['budget'] = Log_result.get('budget', list()) + cost_arr
		Log_result_edges['budget'] = Log_result_edges.get('budget', list()) + cost_arr
		Log_result_step_sel_node['budget'] = Log_result_step_sel_node.get('budget', list()) + range(1,len(sample._track_selected_node)+1)
		Log_result_step_sel_rank['budget'] = Log_result_step_sel_rank.get('budget', list()) + range(1,len(sample._track_selected_node)+1)


		starting_node = -1

	# Create a figure instance
	fig = plt.figure(1, figsize=(9, 6))
	# Create an axes instance
	ax = fig.add_subplot(111)
	# Create the boxplot
	bp = ax.boxplot(sample._data_to_plot)
	# Save the figure
	fig.savefig('./draw/plot/'+dataset+'.png', bbox_inches='tight')

	_mylib.plotLineGraph([sample._chose_deg], log=False, title=dataset)

	print(len(sample._chose_deg), sample._chose_deg[0], sample._chose_deg[1])
	print(len(sample._data_to_plot))

	SaveToFile(Log_result, Log_result_edges, Log_result_step_sel_node, Log_result_step_sel_rank)

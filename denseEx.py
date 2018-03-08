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
import multi_sample as ml

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

		self._starting_wt = 0
		self._wt_den = 3 #0.90 # 3 				# Weight of the densification score
		self._wt_exp = 1 #(1 - self._wt_den)			# Weight of the expansion score

		self._score_den_list = [] 			# List to store densification scores; used only for logging
		self._score_exp_list = [] 			# List to store expansion scores; used only for logging
		self._new_nodes = []			# List of the new nodes observed at each densification
		self._ext_nodes = []
		self._cumulative_new_nodes = [] # Cummulated new nodes
		self._exp_cut_off = 15			# Numbor of expansion candidates
		self._den_cut_off = 100 			# Number of densificaiton candidates
		self._score_den_threshold = 0.1
		self._sample_graph = nx.Graph() # The sample graph
		self._nodes_observed_count = [] # Number of nodes observed in each iteration
		self._avg_deg = 0.
		self._med_deg = 0.
		self._cost_spent = []
		self._nodes_return = []
		self._exp_count = 1
		self._densi_count = 1
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

		self._bandit = {'arms': dict.fromkeys([1,2,3,4]),
				  'score': dict.fromkeys([1,2,3,4], float('inf')),
				  'count': dict.fromkeys([1,2,3,4], 0),
				  'created_at': dict.fromkeys([1,2,3,4], 0),
				  'rewards': defaultdict(list)}

		self._sel_obs_deg = dict()
		self._sel_act_deg = dict()
		self._deg_correlation = []
		self._p_value = []

		self._expected_avg_deg = 0
		self._wt_increment = 10

	def _expansion_random(self, candidate_list):
		print(' == Expansion : cand: {} == '.format(len(candidate_list)))
		degree_observed = self._sample_graph.degree(candidate_list)
		degree_observed_sorted = _mylib.sortDictByValues(degree_observed, reverse=True)
		top_20 = int(.2 * len(degree_observed_sorted))
		sel = random.choice(degree_observed_sorted[top_20:])
		node = sel[0]
		deg = sel[1]
		current = node

		print(' == End Expansion : top-20: {} \t node: {}== '.format(top_20, node))

		return current

	def _expansion(self, candidate_list):

		print(' == Expansion : cand: {} == '.format(len(candidate_list)))
		ecc = nx.eccentricity(self._sample_graph)
		ecc_of_starting_node = ecc[starting_node]

		closed_nodes = self._sample['nodes']['close']
		ecc_open = _mylib.remove_entries_from_dict(list(closed_nodes), ecc)

		max_ecc_val = max(ecc_open.values())
		nodes_with_max_val = [k for k, v in ecc.iteritems() if v == max_ecc_val]

		print(" Max ecc:{} \t start ecc: {} \t {}/{}".format(max_ecc_val, ecc_of_starting_node,
															 len(nodes_with_max_val), len(candidate_list)))
		print("="*10)
		current = random.choice(nodes_with_max_val)

		return current

	def _densification(self, current):
		"""
		Run the densification steps

		Args:
			candidate (str) -- The id of the node to start densification on
		Return:
			list[str] -- List of candidate nodes for expansion
		"""
		print(' --- Start Densification: \t node {}'.format(current))
		# If the candidate is not in the sample, add it
		if current not in self._sample_graph.nodes():
			print('Candiate {} is not in sample .. added'.format(current))
			self._sample_graph.add_node(current)

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		# Initialize densification and expansion scores
		score_den = self._scoreDen(sub_sample)
		score_exp = self._scoreExp(sub_sample)

		prev_score = score_den
		current_node_deg_obs = 0.

		isStop = False
		# Perform densification until one of the conditions is met:
		# 	1. Densification score is less than the expansion score
		# 	2. Budget allocated has run out
		# 	3. There are no more open nodes
		# TODO: Densification switch criteria
		while isStop == False and\
				(self._cost < self._budget and len(sub_sample['nodes']['open']) > 0):

			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			new_nodes = set(nodes) - set(self._sample_graph.nodes())
			# Update
			sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			#print('{} #{} cost:{} \t current node: {} \t| den: {} \t exp:{}|\t cov: {} \t new: {}/{} avg:{}'.format(type, i, self._cost, current,
																				# score_den, score_exp,  self._sample_graph.number_of_nodes(),
																				# 							   len(new_nodes), len(nodes), self._expected_avg_deg))


			# TODO: Densification score, to be changed.
			# Update the densification and expansion scores
			score_den  = self._scoreDen(sub_sample, new_nodes, nodes,
										current_node_deg_obs, score_den)
			score_exp = self._scoreExp(sub_sample, new_nodes, nodes,
									   current_node_deg_obs, score_exp)

			self._densi_count += 1

			if score_den < score_exp or len(sub_sample['nodes']['open']) == 0 :
				print(' Score Den: {} \t Score Exp: {} \t Switch '.format(score_den, score_exp))
				isStop = True
			else:
				# Selecting next node
				den_nodes = self._getDenNodes(list(sub_sample['nodes']['open']))
				degree_observed = self._sample_graph.degree(den_nodes)
				current, current_node_deg_obs = self._densification_node_selection(den_nodes)



		# Update the sample with the sub sample
		self._updateSample(sub_sample)
		print(' --- End Densification --- ')
		# Return list of potential expansion nodes
		return self._getExpNodes()

	def _densification_node_selection(self, candidates):
		degree_observed = self._sample_graph.degree(candidates)

		if len(candidates) > 10:
			min_val = min(degree_observed.values())
			max_val = max(degree_observed.values())
			score_d = dict()
			cc = nx.clustering(self._sample_graph, candidates)

			for k, v in degree_observed.iteritems():
				try:
					score_d[k] = ((v - min_val) / (max_val - min_val)) * (1 - cc[k])
				except ZeroDivisionError:
					score_d[k] = v

			max_score = max(score_d.values())
			nodes_with_max_score = [k for k, v in score_d.iteritems() if v == max_score]
			current = random.choice(nodes_with_max_score)
		else:
			current = list(candidates)[0]

		current_node_obs_deg = degree_observed[current]

		return current, current_node_obs_deg

	def _getDenNodes(self, nodes):
		graph = self._sample_graph

		obs_deg = graph.degree(nodes)
		sorted_list = _mylib.sortDictByValues(obs_deg, reverse=True)
		top_k = int(.20 * len(sorted_list))

		if top_k == 0:
			candidates = obs_deg.keys()
		else:
			candidates = [x[0] for x in sorted_list[:top_k]]
		return candidates

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

	def _scoreDen(self, sub_sample, new_nodes=None, nodes=None, current_node_obs_deg=0, prev=0):
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
			print(' *** [scoreDen] nodes is empty, returns INF.')
			return np.inf

		boundary_nodes = nx.node_boundary(self._sample_graph, sub_sample['nodes']['close'])
		check = set(sub_sample['nodes']['open']).intersection(set(boundary_nodes))


		# TODO: score function
		# Calculate the densification score: deg_new / (deg_true - deg_obs)
		try:
			score = 1.*len(new_nodes) / (len(nodes) - current_node_obs_deg)
		except ZeroDivisionError:
			score = 0.

		# Update weight
		#if current_node_obs_deg >= (self._expected_avg_deg):
		#	self._wt_den += self._wt_increment


		if np.isfinite(prev):
			s = (0.5 * prev) + (self._wt_den * score)
		else:
			s = self._wt_den * score

		return s

	def _scoreExp(self, sub_sample, new_nodes=None, nodes=None, current_node_obs_deg=0, prev=0):
		"""
		Calculate the expansion score

		Args:
			sub_sample(dict) -- The sub sample from current densification
			prev (float) -- The previous expansion score
		"""

		if nodes is None:
			return 0.

		edges_boundary = nx.edge_boundary(self._sample_graph, sub_sample['nodes']['close'])

		# TODO: score function
		# Calculate the densification score
		try:
			score = 1.*(len(nodes) - len(new_nodes) - current_node_obs_deg)  / (len(nodes) - current_node_obs_deg)
		except ZeroDivisionError:
			score = 0.

		s = (0.5 * prev) + (self._wt_exp * score)
		return s

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

	def _bfs_init(self):

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
		while self._cost < self._bfs_count and len(queue) > 0:
			# Select the first node from queue
			current = queue[0]

			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c, _ = self._query.neighbors(current)
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

	def _smooth_init(self):
		current = starting_node
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		sample_nodes = set()

		current_node_deg_obs = 0

		while self._cost < self._bfs_count:
			closed_nodes = sub_sample['nodes']['close']
			nodes, edges, c, _ = self._query.neighbors(current)

			# This is sampling with replacement.
			if current not in closed_nodes:
				sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			print('{} #{} [INIT] cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))

			next_node = random.choice(list(nodes))
			next_node_deg = self._sample_graph.degree(next_node)

			a = (1.*next_node_deg / len(nodes))
			PROB = min(1, a)
			r = random.uniform(0, 1)

			if r < PROB:
				current = next_node
				current_node_deg_obs = self._sample_graph.degree(current)

			# if self._cost > 0:
				sample_nodes.add(current)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

		degree_sampled = self._sample_graph.degree(sample_nodes)
		print('Sampled nodes', len(sample_nodes))

		########## Estimate average degree ##############
		a = 0.
		b = 0.
		c = 0
		for node, deg in degree_sampled.iteritems():
			a += 1.*deg / (deg + c)
			b += 1. / (deg + c)
		self._expected_avg_deg = int(math.ceil(a/b))
		print(' Expected Avg. Deg', self._expected_avg_deg )
		########################################

	def _after_init(self):
		true_g = self._query._graph
		act_deg = true_g.degree()
		act_min_deg = min(act_deg.values())
		act_max_deg = max(act_deg.values())
		average_deg = np.average(np.array(act_deg.values()))

		current_g = self._sample_graph
		close_nodes = self._sample['nodes']['close']
		open_nodes = self._sample['nodes']['open']

		deg = current_g.degree(list(close_nodes))
		min_deg = min(deg.values())
		max_deg = max(deg.values())
		avg = np.average(np.array(deg.values()))
		med = np.median(np.array(deg.values()))
		#self._expected_avg_deg = med

		deg_open = current_g.degree(open_nodes)
		s_deg = _mylib.sortDictByValues(deg_open, reverse=True)
		max_deg_open = s_deg[0][1]

		# Initial weight for densifcation score
		self._wt_den = max_deg / self._expected_avg_deg
		self._starting_wt = self._wt_den


	def _bfs(self, isSnowball=False):

		"""
		Collect the initial nodes through bfs

		Args:
			None
		Return:
			None
		"""
		current = starting_node

		sub_sample = {'edges':set(), 'nodes':{'close':set(), 'open':set()}}
		sub_sample['nodes']['open'].add(current)

		queue = [current]
		current_node_deg_obs = 0

		# Run till bfs budget allocated or no nodes left in queue
		while self._cost < self._budget and len(queue) > 0:
			# Get the neighbors - nodes and edges; and cost associated
			nodes, edges, c, _ = self._query.neighbors(current)

			# Remove the current node from queue
			queue.remove(current)
			next_nodes = set(nodes) - sub_sample['nodes']['close']
			next_nodes = set(next_nodes) - sub_sample['nodes']['open']

			# Exception: put this line here becuase nodes that are added to the query should not be duplicate
			#  Check `next_nodes` first before doing any update.
			sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)



			if isSnowball and len(next_nodes) != 0:
				s_size = int(math.ceil(.5 * len(next_nodes)))
				next_nodes = random.sample(list(next_nodes), s_size)
			queue += list(next_nodes)
			queue = list(set(queue))

			print('{} #{} cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))

			# Select the first node from queue
			current = queue[0]
			current_node_deg_obs = self._sample_graph.degree(current)

		# Updat the sample with the sub sample
		self._updateSample(sub_sample)

	def random_sampling(self):

		current = starting_node

		# Initialize a new sub sample
		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		current_node_deg_obs = 0
		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			print('{} #{} cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))

			# Randomly pick node
			current = random.choice(list(sub_sample['nodes']['open']))
			current_node_deg_obs = self._sample_graph.degree(current)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _random_walk(self):
		current = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		current_node_deg_obs = 0

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)

			# Count and track everthing if selected node is open. Otherwise, move to neighbor
			if current not in sub_sample['nodes']['close']:
				sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			print('{} #{} cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))

			current = random.choice(list(nodes))
			current_node_deg_obs = self._sample_graph.degree(current)

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _cash_spread(self):
		current = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		C = dict()
		H = dict()
		G = 0
		P = dict()

		current_node_deg_obs = 0

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			print('{} #{} cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))

			cost_of_current = C.get(current, 1.)
			H[current] = H.get(current, 0.) + cost_of_current

			G += cost_of_current
			for nb in nodes:
				C[nb] = C.get(nb, 1.) + (cost_of_current / len(nodes))
				P[nb] = (H.get(nb, 0.) + C.get(nb, 1.)) / (G + 1)

			P[current] = 1. * (H.get(current, 0) + C.get(current, 1)) / (G + 1)
			C[current] = 0.

			closed_node = sub_sample['nodes']['close']

			P_open = _mylib.remove_entries_from_dict(closed_node, C)
			max_p = max(P_open.values())

			nodes_with_max_p = [k for k, v in P_open.iteritems() if v == max_p]
			print('node with max p ', len(nodes_with_max_p), len(P), len(P_open))
			current = random.choice(nodes_with_max_p)
			current_node_deg_obs = self._sample_graph.degree(current)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_obs_deg(self):
		current = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		current_node_deg_obs = 0

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			# Update
			sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			print('{} #{} cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))

			candidates = sub_sample['nodes']['open']
			degree_observed = self._sample_graph.degree(candidates)
			max_observed_degree = max(degree_observed.values())
			nodes_with_max_deg = [k for k, v in degree_observed.iteritems() if v == max_observed_degree]

			current = random.choice(nodes_with_max_deg)
			current_node_deg_obs = degree_observed[current]


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _max_obs_page_rank(self):
		current = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		current_node_deg_obs = 0

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:
			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			# Update
			sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			print('{} #{} cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))


			candidates = sub_sample['nodes']['open']
			closed_nodes = set(self._sample_graph.nodes()) - candidates

			observed_page_rank = nx.pagerank(self._sample_graph)
			observed_page_rank_open = _mylib.remove_entries_from_dict(list(closed_nodes), observed_page_rank)
			max_page_rank = max(observed_page_rank_open.values())
			nodes_with_max_page_rank = [k for k, v in observed_page_rank_open.iteritems() if v == max_page_rank]

			current = random.choice(nodes_with_max_page_rank)
			current_node_deg_obs = self._sample_graph.degree()[current]

		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _updateAfterQuery(self, nodes, edges, c, current, current_node_deg_obs, sub_sample):
		# For tracking and logging
		self._count_new_nodes(nodes, current, deg_obs=current_node_deg_obs)

		# Add edges to sub_graph
		for e in edges:
			self._sample_graph.add_edge(e[0], e[1])

		# Update the sub sample
		sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

		# Update the cost
		self._increment_cost(c)

		return sub_sample

	def _count_new_nodes(self, nodes, current, deg_obs=0, score=0):
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

	def _max_excess_deg(self):
		current = starting_node

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)
		sub_sample['nodes']['open'].update(self._sample['nodes']['open'])

		current_node_deg_obs = 0

		while self._cost < self._budget and len(sub_sample['nodes']['open']) > 0:

			close_n = sub_sample['nodes']['close']

			# Query the neighbors of current
			nodes, edges, c, _ = self._query.neighbors(current)
			sub_sample = self._updateAfterQuery(nodes, edges, c, current, current_node_deg_obs, sub_sample)

			print('{} #{} cost:{} \t current node: {} \t node coverage: {}'.format(type, i, self._cost, current,
																				   self._sample_graph.number_of_nodes()))

			# Pick node with the maximum number of excess degree.
			candidates = sub_sample['nodes']['open']
			current = self._get_max_excess_degree_node(candidates)
			current_node_deg_obs = self._sample_graph.degree(current)


		# Update the sample with the sub sample
		self._updateSample(sub_sample)

	def _get_max_excess_degree_node(self, candidates):
		degree_true = self._query._graph.degree(candidates)
		degree_observed = self._sample_graph.degree(candidates)

		degree_excess = Counter(degree_true)
		degree_excess.subtract(Counter(degree_observed))

		max_excess = max(degree_excess.values())
		nodes_with_max_excess_deg = [k for k, v in degree_excess.iteritems() if v == max_excess]

		current = random.choice(nodes_with_max_excess_deg)
		return current

	def generate(self):
			"""
			The main method that calls all the other methods
			"""

			current_list = []

			if self._exp_type == 'denseEx' and self._bfs_count != 0:
				self._smooth_init()
				self._after_init()

			# Sample until budget runs out or thero are no more open nodes
			while self._cost < self._budget:

				# If there are no more nodes in current list, use open nodes from sample
				if len(current_list) < 1:
					current_list = list(self._sample['nodes']['open'])

				# Perform expansion
				if len(current_list) != 0 and self._densi_count != 1:
					self._stage = 'exp'
					self._exp_count += 1
					if self._exp_type == 'oracle':
						current_list = self._getExpNodes()
						current = self._expansion_oracle(current_list)
					elif self._exp_type == 'denseEx':
						#current = self._expansion_random(self._sample['nodes']['open'])
						current = self._expansion(self._sample['nodes']['open'])

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
				elif self._exp_type == 'pagerank':
					self._max_obs_page_rank()
				elif self._exp_type == 'med':
					self._max_excess_deg()
				elif self._exp_type == 'oracle':
					current_list = self._densification_oracle(current)
				elif self._exp_type == 'sb':
					self._bfs(isSnowball=True)
				elif self._exp_type == 'bfs':
					self._bfs()
				elif self._exp_type == 'opic':
					self._cash_spread()
				elif self._exp_type == 'denseEx':
					current_list = self._densification(current)



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
	log.save_to_file(log_file_edge, results_edges)
	log.save_to_file(log_file_order, query_order)
	#log.save_to_file(log_file_rank, rank_order)


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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-task', help='Type of sampling', default='undirected_single')
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-budget', help='Total budget', type=int, default=0)
	parser.add_argument('-percent_b', help='percent budget', type=int, default=0.1)
	parser.add_argument('-bfs_budget', help='Bfs budget', type=int, default=5)
	parser.add_argument('-p_bfs_budget', help='Percent Bfs budget', type=float, default=.015)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)
	parser.add_argument('-log', help='Log file', default='./log/')
	parser.add_argument('-experiment', help='# of experiment', default=10)
	parser.add_argument('-log_interval', help='# of budget interval for logging', type=int, default=10)
	parser.add_argument('-k', help='top k percent', type=int, default=5)
	parser.add_argument('-is_cost', help='take cost in account', type=bool, default=True)
	parser.add_argument('-mode', help='mode', type=int, default=1)
	parser.add_argument('-delimiter', help='csv delimiter', type=str, default=None)
	parser.add_argument('-debug', help='debug mode', type=bool, default=False)

	args = parser.parse_args()

	print(args)

	fname = args.fname
	budget = args.budget
	bfs_budget = args.bfs_budget
	p_bfs_budget = args.p_bfs_budget
	dataset = args.dataset
	log_file = args.log
	k = args.k
	is_cost = args.is_cost
	log_interval = args.log_interval
	mode = args.mode
	delimeter = args.delimiter
	debug = args.debug
	P_BUDGET = args.percent_b

	if mode == 1:
		exp_list = ['med', 'mod', 'rw', 'bfs', 'sb', 'random', 'opic', 'pagerank']
		#exp_list = ['med','mod','rw','denseEx']
	elif mode == 2:
		exp_list = ['med', 'mod', 'rw', 'denseEx']
	elif mode == 3:
		exp_list = ['denseEx']


	print(exp_list)
	Log_result = {}
	Log_result_edges = {}
	Log_result_step_sel_node = {}
	Log_result_step_sel_rank = {}

	Log_result_hyb = {}

	if dataset == None:
		fname = fname.replace('\\', '/')
		f = fname.split('.')[1].split('/')[-1]
		dataset = f

	if args.task == 'undirected_single':
		G = _mylib.read_file(fname)

		print('Original: # nodes', G.number_of_nodes())
		graph = max(nx.connected_component_subgraphs(G), key=len)

		print('LCC: # nodes', graph.number_of_nodes())
		query = query.UndirectedSingleLayer(graph)
		#oracle = oracle.Oracle(graph, dataset)

		if mode != 3:
			log_file_node = log_file + dataset + '_n.txt'
			log_file_edge = log_file + dataset + '_e.txt'
			log_file_order = log_file + dataset + '_order.txt'
			log_file_rank = log_file + dataset + '_rank.txt'
		else:
			log_file_node = log_file + dataset + 'exp_n.txt'
			log_file_edge = log_file + dataset + 'exp_e.txt'
			log_file_order = log_file + dataset + 'exp_order.txt'
			log_file_rank = log_file + dataset + 'exp_rank.txt'

		n = graph.number_of_nodes()

		if budget == 0:
			budget = int(P_BUDGET*n)

		bfs_budget = int(p_bfs_budget*n)


		print('** {} Budget set to {} and {} BFS, n={}'.format(dataset, budget, bfs_budget, n))

	print('-'*10)
	print(nx.info(graph))
	avg_true_deg = np.average(np.array(graph.degree().values()))

	print('-' * 10)

	for i in range(0, int(args.experiment)):
		row = []

		tmp = []
		for type in exp_list:
			sample = UndirectedSingleLayer(query, budget, bfs_budget, type, dataset, log, k, is_cost, log_interval)

			if starting_node == -1:
			    starting_node = sample._query.randomNode()



			# Getting sample
			sample.generate()
			# End getting sample

			cost_arr = Append_Log(sample, type)


		Log_result['budget'] = Log_result.get('budget', list()) + cost_arr
		Log_result_edges['budget'] = Log_result_edges.get('budget', list()) + cost_arr
		Log_result_step_sel_node['budget'] = Log_result_step_sel_node.get('budget', list()) + range(1,len(sample._track_selected_node)+1)
		Log_result_step_sel_rank['budget'] = Log_result_step_sel_rank.get('budget', list()) + range(1,len(sample._track_selected_node)+1)


		starting_node = -1

	if not debug and int(args.experiment) != 0:
		SaveToFile(Log_result, Log_result_edges, Log_result_step_sel_node, Log_result_step_sel_rank)

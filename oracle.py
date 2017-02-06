# Network Sampling: Oracle v 0.1
# Updated on: Aug 10, 2016
from __future__ import division
import networkx as nx
import community
import numpy as np
import cPickle as pickle
import os
import _mylib
import random

class Oracle:
	def __init__(self,G,dataset=None):
		self.graph = G
		com_fname = './data/pickle/communities_{}.pickle'.format(dataset)
		if os.path.isfile(com_fname):
			self.partition = pickle.load(open(com_fname, 'rb'))
		else:
			self.partition = community.best_partition(G)
			pickle.dump(self.partition, open(com_fname, 'wb'))
		self.partitionSet = set(self.partition.values())
		self.membersCount = self.countMembersInCommunity()
		self._diameter, self._center = self.calculateDiameter(self.partition,dataset)
		#self._spread = self.spread(_diameter)
		self._cost_expansion = 0.05
		self._cost_densification = 0.01
		self._communities_selected = {}
	#
	# def spread(self):
	# 	min = min(self._diameter.values())
	# 	max = max(self._diameter.values())
	# 	s = {}
	# 	for k,v in self._diameter.iteritems():
	# 		norm_v = 1.*(v - min) / (max - min)
	# 		s[k] = 1. - norm_v
	# 	return s


	def calculateDiameter(self, partition,dataset):
		G = self.graph

		diameter = {}
		center = {}
		com_fname = 'data/pickle/diameter_{}.pickle'.format(dataset)
		com_fname2 = 'data/pickle/center_{}.pickle'.format(dataset)

		if os.path.isfile(com_fname):
			 return pickle.load(open(com_fname, 'rb')),  pickle.load(open(com_fname2, 'rb'))

		for p in set(partition.values()):
			members = self.getAllNodesInCom(p)
			gg = nx.subgraph(G, members)
			print(" Partition {} {}".format(p, gg.number_of_nodes()))
			g = max(nx.connected_component_subgraphs(gg), key=len)
			print(" Partition' {} {}".format(p, g.number_of_nodes()))
			diameter[p] = nx.diameter(g)
			center[p] = nx.center(g)

		pickle.dump(diameter, open(com_fname, 'wb'))
		pickle.dump(center, open(com_fname2, 'wb'))

		return diameter, center


	def expansion_old(self, candidates_list, observed_list):
		G = self.graph

		best_node = candidates_list[0]
		max_score = 0

		scores = {}

		for candidate_node in candidates_list:
			node_label = self.partition[candidate_node]
			# Get all nodes in the same community as candidate node
			all_nodes = self.getAllNodesInCom(node_label)
			neighbors_set = G.neighbors(candidate_node)

			# Find all unobserved neighbors, (neighbors - observed).
			a = set(neighbors_set).difference(observed_list)
			# Find all nodes that are not in the same community as candidate node, (N - N_c)
			b = set(G.nodes()).difference(all_nodes)
			# Find all the neighbours that are in different community as candidate node
			unobserved = a.intersection(b)

			scores[candidate_node] = 0
			if self.checkListLeadsToGoodCluster(unobserved, observed_list) or True:
				scores[candidate_node] = len(unobserved) * self.clusterScore(candidate_node, observed_list)
				
				if node_label in self._communities_selected:
					scores[candidate_node] = scores[candidate_node]/(self._communities_selected[node_label] + 1)
	
			"""
			# Pick the candidate node that gives the highest unobserved node
			if len(unobserved) > max_unobserved_nodes_count and self.checkListLeadsToGoodCluster(unobserved):
				best_node = candidate_node
				max_unobserved_nodes_count = len(unobserved)
				#test = [self.partition.get(key) for key in unobserved]
				#print 'Best', best_node, unobserved, test
			"""
		for node in scores:
			if scores[node] > max_score:
				best_node = node
				max_score = scores[node]

		if self.partition[best_node] not in self._communities_selected:
			self._communities_selected[self.partition[best_node]] = 1
		else:
			self._communities_selected[self.partition[best_node]] += 1
		
		return best_node, self._cost_expansion * len(candidates_list)

	def expansion(self, candidates_list, observed_list, center_s):
		# TODO: only gain and spread
		g_s = self.calculateCommunityScore_gs(observed_list, center_s)

		# Pick a center node of a community that has the highest score
		g_s_highest = _mylib.sortDictByValues(g_s,reverse=True)[0]
		p = g_s_highest[0]
		p_score = g_s_highest[1]
		center_node = self._center[p][0]
		radius = self._diameter[p] / 2

		# Calculate shortest path from candidate node to center node
		candidate_path = {}
		for candidate in candidates_list:
			try:
				path_length = nx.astar_path_length(self.graph, candidate, center_node)
				if path_length > radius:
					candidate_path[candidate] = path_length - radius
				else:
					candidate_path[candidate] = 0
					break
			except nx.exception.NetworkXNoPath:
				print(" Node not reachable {} {}".format(candidate, center_node))
				continue

		sort_l = _mylib.sortDictByValues(candidate_path)
		best_node = sort_l[0][0]
		dist = sort_l[0][1]
		#print ' Oracle picks ', best_node, ' distance = ', dist
		print " Oracle picks %s distance = %s - com %s score= %s" % (best_node, dist, p, p_score)
		return best_node, self._cost_expansion * len(candidate_path)

	# Calculate score for each community
	def calculateCommunityScore_gs_dist(self, observed_list, center_s):
		gain = self.remainingNodesInCom(observed_list)
		diameter = self._diameter

		score = {}
		W = 0.8
		for com_id, v in gain.iteritems():
			com_center = self._center[com_id][0]
			#path_length = nx.astar_path_length(self.graph, center_s, com_center)
			path_length = nx.dijkstra_path_length(self.graph, center_s, com_center)
			g_s = gain[com_id] * (1. / diameter[com_id])

			try:
				if g_s == 0: score[com_id] = 0
				else: score[com_id] = W * g_s + (1-W) * (1./path_length)
			except ZeroDivisionError:
				score[com_id] = W * g_s + (1 - W)
			print('	Com score {} {} -- left {} hop {}'.format(com_id, score[com_id], gain[com_id], path_length))


		return score

	def calculateCommunityScore_gs(self, observed_list, center_s):
		gain = self.remainingNodesInCom(observed_list)
		diameter = self._diameter

		score = {}
		W = 0.8
		for com_id, v in gain.iteritems():
			com_center = self._center[com_id][0]
			g_s = gain[com_id] * (1. / diameter[com_id])
			score[com_id] = g_s

		return score

	# Calculate how many nodes left unexplored in each community.
	# Score = number of unobs nodes / |V|
	def remainingNodesInCom(self, observed_list):
		partition = self.partition
		score = {}
		for p in set(partition.values()):
			members_in_p = self.getAllNodesInCom(p)
			unobs_members = set(members_in_p) - set(observed_list)
			score[p] = 1. * (len(unobs_members)) / self.graph.number_of_nodes()
			#score[p] = 1. * (len(unobs_members)) / len(members_in_p)
		return score

	def densification(self, candidates_list, observed_list):
		G = self.graph

		d = {}

		for candidate_node in candidates_list:
			neighbors_set = G.neighbors(candidate_node)
			# Find all unobservd neighbors that are in the same community as candidate node
			#unobserved = set(neighbors_set).difference(observed_list).intersection(all_nodes)
			unobserved = set(neighbors_set).difference(observed_list)
			d[candidate_node] = len(unobserved)

		max_val = max(d.values())
		c = list(_mylib.get_members_from_com(max_val, d))

		#print('max ', max_val, len(c))

		return random.choice(c), self._cost_densification * len(candidates_list)

	# Check whether a node is part of a satisfied cluster or not, depends on the threshold.
	def inCluster(self, node, observed_list, THRESHOLD=.05):
		isInCluster = False
		totalNodes = len(self.partition.keys()) - len(observed_list)
		com = self.partition[node]
		comSize = self.membersCount[com]

		if (1.*comSize / totalNodes) >= THRESHOLD:
			isInCluster = True

		return isInCluster

	def clusterScore(self, node, observed_list):
		totalNodes = len(self.partition.keys()) - len(observed_list)
		com = self.partition[node]
		comSize = self.membersCount[com]

		return comSize/totalNodes

	def countMembersInCommunity(self):
		partitionSet = set(self.partition.values())

		membersCount = {}
		for p in partitionSet:
			nodes = self.getAllNodesInCom(p)
			membersCount[p] = len(nodes)
		return membersCount

	def getAllNodesInCom(self, p):
		valueList = np.array(self.partition.values())
		indices = np.argwhere(valueList == p)
		indices = indices.reshape(len(indices))

		keyList = np.array(self.partition.keys())

		nodes = keyList[indices]
		return nodes

		# Check list whether it contains nodes that reside in a good cluster

	def checkListLeadsToGoodCluster(self, nodes_list, observed_list):
		for node in nodes_list:
			if self.inCluster(node, observed_list):
				return True

		return False

	# TODO: Fix this
	def costToNearestCluster(self, candidates_list, samples_list):
		partitionSet = set(self.partition.values())
		for p in partitionSet:
			nodes = self.getAllNodesInCom(p)
			print nodes, p

		return 0

	def get_max_unobs_nodes(self,nodes_list, observed_list):
		best_count = 0
		best_node = None

		for node in nodes_list:
			nodes = self.graph.neighbors(node)

			# Focus on the unobserved NODE instead of unobserved DEGREE
			unobserved_nodes_count = len(set(nodes) - set(observed_list))

			if unobserved_nodes_count > best_count or best_count == 0:
				best_count = unobserved_nodes_count
				best_node = node

		return best_node, self._cost_expansion * len(nodes_list)


if __name__ == '__main__':
	# Toy Example
	#G = nx.karate_club_graph()
	G = nx.read_edgelist('com-dblp.ungraph.txt')
	o = Oracle(G)

	# print o.inCluster(8)
	# print o.expansion([2,3,8],[1,2,3,8])
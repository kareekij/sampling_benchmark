from __future__ import division, print_function
import numpy as np
import networkx as nx

import _mylib
import community
import pickle
import os
import argparse
import scipy.stats as stats
import random

class MultiLayersNetwork(object):
	def __init__(self):
		super(MultiLayersNetwork, self).__init__()
		self._layers_count = 0
		self._layers = {'expensive': list(), 'cheap': list(), 'compose': nx.Graph(),  'compose_c': nx.Graph()}
		self._partition = {'expensive': list(), 'cheap': list() }
		self._query_cost = []
		self._cost_neighbor_exp = 1
		self._cost_neighbor_chp = 0.2
		self._cost = 0
		self._budget = 50
		self._sample_graph = nx.Graph()
		self._fname = ""

	def _get_cheap_layer(self, id=0):
		return self._layers['cheap'][id]

	def init_layer(self, layers, folder):
		self._layers_count = len(layers)
		self._fname = folder

		self._layers['expensive'] = [layers[0]]
		self._layers['cheap'] += layers[1:]

		for i, layer in enumerate(layers):
			p = self.find_community(layer, i)
			if i == 0:
				self._partition['expensive'] = [p]
			else:
				self._partition['cheap'].append(p)

				# Merge cheaper layers to one single network
				self._layers['compose'] = nx.compose(self._layers['compose'], layer)

		nodes_exp = set(self._layers['expensive'][0].nodes())
		nodes_cmp = set(self._layers['compose'].nodes())
		common_n = nodes_exp.intersection(nodes_cmp)
		new_g_common_nodes = self._layers['compose'].subgraph(common_n)

		giant = max(nx.connected_component_subgraphs(new_g_common_nodes), key=len)


		self._layers['compose_c'] = giant


	def neighbors(self, node, type="exp", chp_id=0):
		if type == 'exp':
			nodes = self._layers['expensive'][0].neighbors(node)
			edges = [(node, n) for n in nodes]
			cost = self._cost_neighbor_exp
		elif type == 'chp':
			nodes = self._layers['cheap'][chp_id].neighbors(node)
			edges = [(node, n) for n in nodes]
			cost = self._cost_neighbor_chp

		return set(nodes), set(edges), cost

	def find_community(self, G, i):
		com_fname = self._fname +'com/com_layer_{}.pickle'.format(i)
		if os.path.isfile(com_fname):
			p = pickle.load(open(com_fname, 'rb'))
		else:
			p = community.best_partition(G)
			pickle.dump(p, open(com_fname, 'wb'))
		return p

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
				nodes.difference(sub_sample['nodes']['close']))
		except KeyError:
			print('		Err:')


		return sub_sample

	def _get_local_community(self, com_id):
		expensive_layer = self._layers['expensive'][0]
		cheaper_layer = self._layers['cheap'][0]

		p = multi_layers._partition['expensive'][0]
		members = _mylib.get_members_from_com(com_id, p)

		current = random.choice(list(members))


		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		stop = False
		local_com = set()
		queue = [current]
		while self._cost < self._budget:
			nodes, edges, c = self.neighbors(current)
			self._cost += c

			queue.remove(current)
			nodes = nodes.difference(sub_sample['nodes']['close'])
			nodes = nodes.difference(sub_sample['nodes']['open'])
			queue += list(nodes)
			queue = list(set(queue))

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			if self._cost < 5:
				current = queue[0]
			else:
				local_com = sub_sample['nodes']['close']
				current = self._select_node_from_cheaper(sub_sample['nodes']['open'], local_com)

		local_com = sub_sample['nodes']['close']

		members = _mylib.get_members_from_com(com_id, p)
		print('Original Member Count:', len(members))

		precision = len(set(local_com).intersection(set(members))) / len(local_com)
		recall = len(set(local_com).intersection(set(members))) / len(members)
		f1 = (2 * precision * recall) / (precision + recall)

		print('Precision {} , Recall {}, F1 {}, Cost {}'.format(precision, recall, f1, self._cost))

	def _get_pagerank(self, cheaper_layer,i):
		pr_fname = folder + '/com/pr_cheap_{}.pickle'.format(i)
		if os.path.isfile(pr_fname):
			pr = pickle.load(open(pr_fname, 'rb'))
		else:
			pr = nx.pagerank(cheaper_layer)
			pickle.dump(pr, open(pr_fname, 'wb'))
		return pr

	def _multi_test(self, com_id):
		cheap_id = 0
		expensive_layer = self._layers['expensive'][0]
		cheaper_layer = self._layers['cheap'][cheap_id]

		p_exp = multi_layers._partition['expensive'][0]
		p_chp = multi_layers._partition['cheap'][cheap_id]

		members_exp = _mylib.get_members_from_com(com_id, p_exp)
		print("GT", len(members_exp))
		# select seed
		current = random.choice(list(members_exp))
		start_node = current
		seed_com_in_cheap = p_chp[current]
		members = _mylib.get_members_from_com(seed_com_in_cheap, p_chp)

		adj_coms = set()

		for node in members:
			nbs = cheaper_layer.neighbors(node)
			for nb in nbs:
				c = p_chp[nb]
				adj_coms.add(c)

		print(adj_coms)
		pr = self._get_pagerank(cheaper_layer, cheaper_layer)
		pr_keys = np.array(pr.keys())
		pr_vals = np.array(pr.values())

		sum = 0
		high_cen_cheap = []
		high_cen_cheap.append(current)
		for c in list(adj_coms):
			members = _mylib.get_members_from_com(c, p_chp)
			sum += (len(members))

			# Get index of all candidate nodes
			ix = np.in1d(pr_keys.ravel(), members).reshape(pr_keys.shape)
			max_val = np.amax(pr_vals[np.where(ix)])
			max_val_index = np.where(pr_vals == max_val)
			sel_node = random.choice(list(pr_keys[max_val_index]))
			high_cen_cheap.append(sel_node)
			#print(c, sel_node, pr[sel_node])


		print(high_cen_cheap)

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		#sub_sample['nodes']['open'].update(high_cen_cheap)

		stop = False
		local_com = set()
		queue = list(high_cen_cheap)

		EQUAL_BUDGET = 10
		for seed in high_cen_cheap:
			current = seed
			sub_sample['nodes']['open'].add(current)
			print('Seed', seed)
			while self._cost < EQUAL_BUDGET and current not in sub_sample['nodes']['close']: #self._budget:
				#current = queue[0]
				nodes, edges, c = self.neighbors(current)
				self._cost += c

				#queue.remove(current)
				#nodes = nodes.difference(sub_sample['nodes']['close'])
				#nodes = nodes.difference(sub_sample['nodes']['open'])
				#queue += list(nodes)
				#queue = list(set(queue))

				# Update the sub sample
				sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

				for e in edges:
					self._sample_graph.add_edge(e[0], e[1])

				candidates = sub_sample['nodes']['open']
				degree_observed = self._sample_graph.degree(candidates)
				degree_true = expensive_layer.degree(candidates)
				mutual_score = {}
				for k, v in degree_true.iteritems():
					mutual_score[k] = degree_observed[k] / degree_true[k]

				degree_observed_sorted = _mylib.sortDictByValues(mutual_score, reverse=True)
				current = degree_observed_sorted[0][0]

		#local_com = self._sample_graph

		# Refine
		current_g = self._sample_graph
		prev_r = score_r = 0
		local_com = set()
		current = start_node

		candidates = set()
		while prev_r <= score_r:
			nbs = current_g.neighbors(current)
			local_com.add(current)
			nbs = set(nbs) - set(local_com)
			candidates.update(set(nbs))

			degree = current_g.degree(list(candidates))
			degree_observed_sorted = _mylib.sortDictByValues(degree, reverse=True)
			current = degree_observed_sorted[0][0]
			candidates.remove(current)

			prev_r = score_r
			score_r = self._local_r(current_g, local_com)

			print('Prev {}, Current {}, {}'.format(prev_r, score_r, len(local_com)))
			if len(candidates) == 0:
				break




		# End
		print('Original Member Count:', len(members_exp))
		print('Found:', len(local_com))

		precision = len(set(local_com).intersection(set(members_exp))) / len(local_com)
		recall = len(set(local_com).intersection(set(members_exp))) / len(members_exp)
		f1 = (2 * precision * recall) / (precision + recall)

		print('Precision {} , Recall {}, F1 {}, Cost {}'.format(precision, recall, f1, self._cost))

	def _local_r(self, g, close_n):
		com = g.subgraph(close_n)

		deg_all = g.degree(close_n)
		deg_count = np.sum(np.array(deg_all.values()))

		internal_edge_count = com.number_of_edges()
		external_edge_count = deg_count - (2*internal_edge_count)

		return (internal_edge_count / (external_edge_count))


	def _select_node_from_cheaper(self, open_nodes, local_com):
		cheaper_layer = self._layers['cheap'][0]
		score = {}

		#pr = self._get_page_rank() #nx.pagerank(cheaper_layer)

		for node in open_nodes:
			nbs = cheaper_layer.neighbors(node)
			score[node] = set(nbs) & set(local_com)
			#score[node] = pr[node]

		sort_score = _mylib.sortDictByValues(score, reverse=True)
		return sort_score[0][0]

	def _get_page_rank(self,i=1):
		cheaper_layer = self._layers['cheap'][0]

		com_fname = folder + '/com/pr_layer_{}.pickle'.format(i)
		if os.path.isfile(com_fname):
			pr = pickle.load(open(com_fname, 'rb'))
		else:
			pr = nx.pagerank(cheaper_layer)
			pickle.dump(pr, open(com_fname, 'wb'))
		return pr

	def _mutual_friend_crawling(self, com_id):
		expensive_layer = self._layers['expensive'][0]
		cheaper_layer = self._layers['cheap'][0]

		p = multi_layers._partition['expensive'][0]
		members = _mylib.get_members_from_com(com_id, p)

		current = random.choice(list(members))

		sub_sample = {'edges': set(), 'nodes': {'close': set(), 'open': set()}}
		sub_sample['nodes']['open'].add(current)

		stop = False
		local_com = set()

		while self._cost < self._budget:
			nodes, edges, c = self.neighbors(current)
			self._cost += c

			for e in edges:
				self._sample_graph.add_edge(e[0], e[1])

			# Update the sub sample
			sub_sample = self._updateSubSample(sub_sample, nodes, edges, current)

			candidates = sub_sample['nodes']['open']
			degree_observed = self._sample_graph.degree(candidates)
			degree_true = expensive_layer.degree(candidates)
			mutual_score = {}
			for k, v in degree_true.iteritems():
				mutual_score[k] = degree_observed[k] / degree_true[k]

			degree_observed_sorted = _mylib.sortDictByValues(mutual_score, reverse=True)
			current = degree_observed_sorted[0][0]

		local_com = sub_sample['nodes']['close']

		members = _mylib.get_members_from_com(com_id, p)
		print('Original Member Count:', len(members))

		precision = len(set(local_com).intersection(set(members))) / len(local_com)
		recall = len(set(local_com).intersection(set(members))) / len(members)
		f1 = (2 * precision * recall) / (precision + recall)

		print('Precision {} , Recall {}, F1 {}, Cost {}'.format(precision, recall, f1, self._cost))

	def run(self, exp, com_id):

		if exp == 'multi':
			#self._get_local_community(com_id)
			self._multi_test(com_id)
		elif exp == 'mod':
			self._mutual_friend_crawling(com_id)


def get_rank_correlation(d_1, d_2, k=.5):
	cutoff = int(k * len(d_1))
	d_1_sorted = _mylib.sortDictByValues(d_1,reverse=True)

	l_1 = []
	l_2 = []
	for count, d in enumerate(d_1_sorted):
		id = d[0]
		val = d[1]

		l_1.append(val)
		l_2.append(d_2[id])

		if count == cutoff:
			#print(count)
			break

	tau, p_value = stats.kendalltau(l_1, l_2)
	print(k, round(tau,4), round(p_value,4))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder', help='reddit folder', default='./data/twitter-common/')
	args = parser.parse_args()

	folder = args.folder
	print('Accessing folder', folder)

	Graph_Layers = []
	for file in os.listdir(folder):
		path = folder + file
		if os.path.isfile(path) and file != '.DS_Store':
			G = nx.Graph()
			G = _mylib.read_file(path)
			Graph_Layers.append(G.copy())
			print('		nodes: {} edges: {}'.format(G.number_of_nodes(), G.number_of_edges()))
			deg = G.degree()


	print('-- Common nodes/edges --')

	idx = 0

	nodes = set(Graph_Layers[idx].nodes())
	edges = set(Graph_Layers[idx].edges())
	g_1 = Graph_Layers[idx]

	compose_g = nx.Graph()
	for i, g in enumerate(Graph_Layers):
		g_nodes = set(g.nodes())
		g_edges = set(g.edges())
		common_nodes = nodes.intersection(g_nodes)
		print(i, 'common nodes/edges', len(common_nodes), len(edges.intersection(g_edges)))


		if i != 0:
			compose_g = nx.compose(compose_g, g, name='merge graph')

		# deg_1 = g_1.degree(list(common_nodes))
		# deg_2 = g.degree(list(common_nodes))
		#
		# get_rank_correlation(deg_2, deg_1, k=.15)

	nodes_exp = nodes
	nodes_chp = set(compose_g.nodes())
	common_nodes = nodes.intersection(nodes_chp)


	print('-'*10)
	print('Common nodes', len(common_nodes))

	deg_common_e = g_1.degree(common_nodes)
	deg_common_c = compose_g.degree(common_nodes)

	sub_compose = compose_g.subgraph(common_nodes)

	print(nx.info(sub_compose))

	#_mylib.degreeHist(sub_compose.degree().values())
	deg_common_c = sub_compose.degree()

	print(nx.number_connected_components(sub_compose))
	giant = max(nx.connected_component_subgraphs(sub_compose), key=len)
	print(nx.info(giant))


	#sorted_deg_e = _mylib.sortDictByValues(deg_common_e, reverse=False)

	# for k in [.5,.4,.3,.2,.1, 0.05]:
	# 	get_rank_correlation(deg_common_e, deg_common_c, k=k)
	#
	# print('-'*10)
	# for k in [.5,.4,.3,.2,.1, 0.05]:
	# 	get_rank_correlation(deg_common_c, deg_common_e, k=k)


	# l = list()
	# k = .01
	# sorted_deg_c = _mylib.sortDictByValues(deg_common_c, reverse=True)
	# top_k = int(k*len(sorted_deg_c))
	# print(top_k)
	# for t in sorted_deg_c[:top_k]:
	# 	k = t[0]
	# 	l.append(k)
	#
	# deg_top_c = compose_g.degree(l)
	# deg_top_e = g_1.degree(l)




	#_mylib.degreeHist(deg_common_c.values())




	#_mylib.degreeHist(deg_common_e.values())
	#_mylib.degreeHist_2([deg_top_e.values(), deg_top_c.values()], legend=['exp','chp'] )

	#_mylib.degreeHist(compose_g.degree().values())

	# nodes_4 = set(Graph_Layers[4].nodes())
	# nodes_5 = set(Graph_Layers[5].nodes())
	#
	# c_nodes = nodes_4.intersection(nodes_5)
	# print(len(c_nodes))


	# Initialize
	# multi_layers = MultiLayersNetwork()
	# multi_layers.init_layer(Graph_Layers)
	#
	# exp_layer = multi_layers._layers['expensive'][0]
	# chp_1_layer = multi_layers._layers['cheap'][0]
	# chp_2_layer = multi_layers._layers['cheap'][1]
	#
	# p_1 = multi_layers._partition['expensive'][0]

	# p_2 = multi_layers._partition['cheap'][0]
	# p_3 = multi_layers._partition['cheap'][1]
	#
	# for p in set(p_2.values()):
	# 	members_1 = _mylib.get_members_from_com(p, p_2)
	# 	if len(members_1) > 100:
	# 		for pp in set(p_1.values()):
	# 			members_2 = _mylib.get_members_from_com(pp, p_1)
	#
	# 			if len(members_1) > len(members_2):
	#
	# 				score = len(set(members_2) & set(members_1)) / len(members_1)
	#
	# 				if score > 0:
	# 					print(round(score, 4), len(members_1), len(members_2))



	#
	# exp_list = ['multi']
	# for type in exp_list:
	# 	multi_layers = MultiLayersNetwork()
	# 	multi_layers.init_layer(Graph_Layers)
	#
	# 	multi_layers.run(exp=type, com_id=2)


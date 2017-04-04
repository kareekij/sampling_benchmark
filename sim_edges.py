from __future__ import division, print_function
import networkx as nx
import numpy as np
import _mylib
import csv
import query
import argparse
import log
import os
import community
import pickle

def read_file_step(fn):
	# 0: random walk
	# 1: random
	# 3: bfs
	# 4: sb
	# 5: mod

	file = open(fn, "r")
	a = []
	with open(fn, 'rb') as f:
		reader = csv.reader(f, delimiter=',')
		for i, row in enumerate(reader):
			if i != 0:
				t = ([int(x.replace('u', '').replace('\'', '').replace(' ', '')) for x in row])
				a.append(t)

	a = np.array(a).transpose()
	return a

def get_line(a, set=0, id=0):
	offset = int(max(a[2].tolist()))

	print(' 	offset:', offset)
	start = set*offset
	end = (set*offset + offset)-1
	#print('		start {} : end {}'.format(start, end))

	r = (a[id][start:end+1])

	if len(r) == 0:
		print('No data')

	return r, offset

def SaveToFile(results):
	log_file = './log/' + dataset + '_edges.txt'

	log.save_to_file(log_file, results)

def simmulate(steps):
	com_steps = []
	new_node_steps = []
	ratio_found_steps = []
	# Start simulating.

	for i, step in enumerate(steps):
		# Track current community
		cur_com = p[step]
		com_steps.append(cur_com)

		nodes, edges, c = query.neighbors(str(step))
		new_nodes = set(nodes).difference(sample_graph.nodes())
		new_node_steps.append(len(new_nodes))

		for e in edges:
			sample_graph.add_edge(e[0], e[1])

		members = members_dict[cur_com]
		members_found = set(sample_graph.nodes()).intersection(members)
		ratio_found = len(members_found) / len(members)
		print(i, ratio_found, len(members_found))
		ratio_found_steps.append(ratio_found)

	return sample_graph, com_steps, new_node_steps, ratio_found_steps

def get_communities(G, dataset):
	com_fname = './data/pickle/communities_{}.pickle'.format(dataset)
	if os.path.isfile(com_fname):
		p = pickle.load(open(com_fname, 'rb'))
	else:
		p = community.best_partition(G)
		pickle.dump(p, open(com_fname, 'wb'))

	return p

def label_reordering(com_steps):
	label_track = {}
	current_label = 1
	new_com_steps = []

	for com in com_steps:
		if com not in label_track:
			label_track[com] = current_label
			current_label += 1
		new_com_steps.append(label_track[com])

	return new_com_steps

def write_to_file(fn, com_steps, new_node_steps, ratio_found_steps, trial):
	size = len(com_steps)
	steps = range(1, size+1)


	if not os.path.isfile(fn):
		f = open(fn, 'a')
		print('step, label, new_nodes, ratio, trial', file=f)
		f.close()

	f = open(fn, 'a')
	for i in range(0, size):
		print('{}, {}, {}, {}, {}'.format(steps[i], com_steps[i], new_node_steps[i], ratio_found_steps[i], trial) , file=f)

def get_members(graph, p):
	d = {}
	for p_label in set(p.values()):
		members = _mylib.get_members_from_com(p_label,p)
		d[p_label] = members.tolist()

	return d

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)
	parser.add_argument('-type', help='sampling type', default=None)

	args = parser.parse_args()
	fname = args.fname
	type = args.type
	dataset = args.dataset

	if dataset == None:
		f = fname.split('.')[1].split('/')[-1]
		dataset = f


	G = _mylib.read_file(fname)

	graph = max(nx.connected_component_subgraphs(G), key=len)
	p = get_communities(graph, dataset)
	members_dict = get_members(graph,p)

	query = query.UndirectedSingleLayer(G)



	Log_result = {}
	fn = './log/' + dataset + '_order.txt'
	print(fn)

	print(dataset, fn)

	a = read_file_step(fn)

	print(G.number_of_nodes())
	print(graph.number_of_nodes())

	if type == 'mod':
		idx = 5
	elif type == 'rw':
		idx = 0

	to_log = []
	to_log.append(dataset)

	#for id in [0,1,3,4,5]:
	# 0: rw, 5: mod
	for id in [idx]:

		nodes_count_all = []
		edges_count_all = []
		for trial in range(0, 10):
			print('Running .. method: {} trial: {}'.format(id, trial+1))

			steps, budget = get_line(a, set=trial, id=id)
			cost = 0
			sample_graph = nx.Graph()

			queried_nodes = (np.array(steps, dtype=str).tolist())
			sample_graph, com_steps, new_node_steps, ratio_found_steps = simmulate(queried_nodes)

			nodes_count = sample_graph.number_of_nodes()
			edges_count = sample_graph.number_of_edges()

			print('	nodes: {} edge: {}'.format(nodes_count, edges_count))

			# nodes_count_all.append(nodes_count)
			# edges_count_all.append(edges_count)
			#
			com_steps = label_reordering(com_steps)

			write_to_file('./log/com-step/'+dataset+'-com-step-'+ type +'.txt',com_steps, new_node_steps, ratio_found_steps, trial)








	# 		if id == 0:
	# 			if 'budget' not in Log_result:
	# 				b = [x for x in range(1, int(budget) + 1)]
	# 				Log_result['budget'] = list(b)
	# 			else:
	# 				Log_result['budget'] += list(b)
	# 				print(' log', len(Log_result['budget']), len(b))
	#
	# 	if id == 0: name = 'rw'
	# 	elif id == 1: name = 'random'
	# 	elif id == 3: name = 'bfs'
	# 	elif id == 4: name = 'sb'
	# 	elif id == 5: name = 'mod'
	#
	# 	Log_result[name] = track_edges
	#
	#
	# SaveToFile(Log_result)



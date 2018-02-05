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
	header = dict()
	with open(fn, 'rb') as f:
		reader = csv.reader(f, delimiter=',')
		for i, row in enumerate(reader):
			if i != 0:
				t = ([int(x.replace('u', '').replace('\'', '').replace(' ', '')) for x in row])
				a.append(t)
			else:
				for idx, x in enumerate(row):
					header[x.replace(' ','')] = idx

	data = np.array(a).transpose()
	return data, header

def get_line(a, set=0, id=0):
	budget_id = header['budget']
	offset = int(max(a[budget_id].tolist()))

	print(' 	offset: {} length: {}'.format(offset, len(a[0])))
	start = set*offset
	end = (set*offset + offset)-1
	print('		start {} : end {}'.format(start, end))

	r = (a[id][start:end+1])

	if len(r) == 0:
		print('No data')

	return r, offset

def SaveToFile(results):
	log_file = './log/' + dataset + '_edges.txt'

	log.save_to_file(log_file, results)

def simulate(steps):
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

def simulate_new_nodes(steps):
	com_steps = []
	new_node_steps = []
	ratio_found_steps = []

	# Start simulating.
	for i, step in enumerate(steps):
		# Track current community
		#cur_com = p[step]


		nodes, edges, c = query.neighbors(str(step))
		new_nodes = set(nodes).difference(sample_graph.nodes())


		for e in edges:
			sample_graph.add_edge(e[0], e[1])

		if i % 10 == 0:
			#com_steps.append(cur_com)
			new_node_steps.append(len(new_nodes))


	return sample_graph, new_node_steps


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


def write_to_file_nn(fn, com_steps, new_node_steps, trial):
	size = len(com_steps)
	steps = range(1, size+1)


	if not os.path.isfile(fn):
		f = open(fn, 'a')
		print('window, label, new_nodes, trial', file=f)
		f.close()

	f = open(fn, 'a')
	for i in range(0, size):
		print('{}, {}, {}, {}'.format(steps[i], com_steps[i], new_node_steps[i], trial) , file=f)


def get_members(graph, p):
	d = {}
	for p_label in set(p.values()):
		members = _mylib.get_members_from_com(p_label,p)
		d[p_label] = members.tolist()
	return d

def save_sample(sample_graph, output):
	if not os.path.exists(save_path):
		os.makedirs(save_path)


	pickle.dump(sample_graph, open(output, 'wb'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
    parser.add_argument('n', help='total nodes', type=int)
    parser.add_argument('m', help='total edges', type=int)


	args = parser.parse_args()
	fname = args.fname
    total_nodes = args.n
    total_edges = args.m

	f = fname.split('.')[1].split('/')[-1]
	dataset = f

	G = _mylib.read_file(fname)
	query = query.UndirectedSingleLayer(G)

	PATH = '/Users/Katchaguy/Google Drive/results/imc2017/'
	#PATH = "E:\\Works\\OneDrive\\PhD\\Research\\results\\imc2017\\"
	save_path = "./data-control-real/" + dataset + "/"

	fn = PATH + 'realworld' + '/' + dataset + '_order.txt'

	ALGO_LIST = ['bfs', 'mod', 'rw']

	Log_result = {}

	print('Dataset: {} Path: {} '.format(dataset, fn))

	data, header = read_file_step(fn)

	for algo in ALGO_LIST:
		algo_id = header[algo]
		sample_graph = nx.Graph()
		for trial in range(0, 10):
			output = save_path + algo + "_" + str(trial) + ".pickle"

			if os.path.isfile(output):
				print('File {} exists'.format(output))
				continue

			print('Running .. method: {} {} trial: {}'.format(algo, algo_id, trial + 1))
			steps, budget = get_line(data, set=trial, id=algo_id)
			cost = 0
			sample_graph = nx.Graph()

			queried_nodes = (np.array(steps, dtype=str).tolist())

			sample_graph, new_node_steps = simulate_new_nodes(queried_nodes)

			nodes_count = sample_graph.number_of_nodes()
			edges_count = sample_graph.number_of_edges()
            nodes_count_norm = nodes_count / total_nodes
            edges_count_norm = edges_count / total_edges

			print('	nodes: {} edge: {}'.format(nodes_count, edges_count))
            print(' % nodes: {} edges:{}'.format(nodes_count_norm, edges_count_norm))
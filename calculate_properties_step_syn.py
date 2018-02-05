from __future__ import division, print_function
import networkx as nx
import numpy as np
import _mylib
import csv
import query as q
import argparse
import log
import os
import community
import pickle
from scipy import stats
import sys

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
	new_node_steps = []
	closed_nodes = set()

	# Start simulating.
	for i, step in enumerate(steps):
		# Track current community
		#cur_com = p[step]

		nodes, edges, c = query.neighbors(str(step))
		new_nodes = set(nodes).difference(sample_graph.nodes())
		closed_nodes.add(str(step))


		for e in edges:
			sample_graph.add_edge(e[0], e[1])

		if i % STEP_INTERVAL == 0:
			log_step = i
			new_node_steps.append(len(new_nodes))
			#################################################
			deg_S = sample_graph.degree()
			avg_deg_S_all = np.average(np.array(deg_S.values()))
			try:
				avg_cc_S_all = nx.average_clustering(sample_graph)
			except ZeroDivisionError:
				avg_cc_S_all = -1.
				continue
			D_all, p_value_all = stats.ks_2samp(deg_G.values(), deg_S.values())

			#################################################
			deg_S = sample_graph.degree(closed_nodes)
			avg_deg_S_c = np.average(np.array(deg_S.values()))
			try:

				avg_cc_S_c = nx.average_clustering(sample_graph, nodes=closed_nodes)
			except ZeroDivisionError:
				avg_cc_S_c = -1.
				continue
			D_c, p_value_c = stats.ks_2samp(deg_G.values(), deg_S.values())

			#################################################
			with open(LOG_OUTPUT_FILE, 'a') as csvfile:
				wt = csv.writer(csvfile, delimiter=' ')
				wt.writerow([dataset, algo, log_step, trial,
							 avg_deg_S_all, avg_cc_S_all, D_all, p_value_all,
							 avg_deg_S_c, avg_cc_S_c, D_c, p_value_c])

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
	print(' > Save to ', output)
	if not os.path.exists(save_path):
		os.makedirs(save_path)


	pickle.dump(sample_graph, open(output, 'wb'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)
	parser.add_argument('-type', help='sampling type', default=None)

	args = parser.parse_args()
	fname = args.fname
	type = args.type
	dataset = args.dataset

	dataset = fname

	# Number of steps before calculating properties
	STEP_INTERVAL = 10

	# Path that contains input graph, using for simulating the queries
	INPUT_G_PATH = '/Users/Katchaguy/Google Drive/datasets/syn-lfr/gen-mix-0.1/'
	# Path that contains log files for reconstructing the sample graph.
	LOG_PATH = '/Users/Katchaguy/Google Drive/results/imc2017/log-syn-mix-low/'
	# Output path where the reconstructed graphs will go to.
	LOG_OUTPUT_PATH = './log_properties/'
	LOG_OUTPUT_FILE = LOG_OUTPUT_PATH + './syn_properties.txt'
	# Specific file that contains the querying order.
	#fn = LOG_PATH + 'log-syn-mix-low' + '/' + dataset + '_order.txt'
	fn = LOG_PATH + dataset + '_order.txt'

	ALGO_LIST = ['bfs', 'mod', 'rw']

	Log_result = {}

	if not os.path.exists(INPUT_G_PATH + fname):
		print(' ** No {} .. skip'.format(INPUT_G_PATH + fname))
		sys.exit()


	if not os.path.isfile(LOG_OUTPUT_FILE):
		print('Creating .. {}'.format(LOG_OUTPUT_FILE))

		with open(LOG_OUTPUT_FILE, 'wb') as csvfile:
			wt = csv.writer(csvfile, delimiter=' ')
			wt.writerow(['dataset','type', 'step', 'trial',
						 'avg.deg.all','avg.cc.all','d.all','p.all',
						 'avg.deg.c', 'avg.cc.c','d.c','p.c'])


	print('Dataset: {} Reading File: {} '.format(dataset, fn))
	# Read the file that contains the quering order.
	data, header = read_file_step(fn)

	# Iterate each of the specified algorithms
	for algo in ALGO_LIST:
		algo_id = header[algo]
		sample_graph = nx.Graph()

		# Construct graph from each recorded trial
		# Each trail comes from different synthetic graph,
		# ** Beware to read actual graph every iteration
		for trial in range(0, 10):
			# Read original graph for quering purpose
			file_G = INPUT_G_PATH + fname + '/' + str(trial+1) + '/network.dat'
			G = nx.Graph()
			G = _mylib.read_file(file_G)
			query = q.UndirectedSingleLayer(G)

			deg_G = G.degree()
			avg_deg_G = np.average(np.array(deg_G.values()))
			#med_deg_G = np.median(np.array(deg_G.values()))
			avg_cc_G = nx.average_clustering(G)
			print('		A: {} {} '.format(avg_deg_G, avg_cc_G))
			with open(LOG_OUTPUT_FILE, 'a') as csvfile:
				wt = csv.writer(csvfile, delimiter=' ')
				wt.writerow([dataset, algo, -1, trial,
							 avg_deg_G, avg_cc_G, 0., 0.,
							 0.,0.,0.,0.])
			# End original graph

			# Start running
			print('Running .. method: {} {} trial: {}'.format(algo, algo_id, trial + 1))
			steps, budget = get_line(data, set=trial, id=algo_id)
			cost = 0
			sample_graph = nx.Graph()

			queried_nodes = (np.array(steps, dtype=str).tolist())

			sample_graph, new_node_steps = simulate_new_nodes(queried_nodes)

			nodes_count = sample_graph.number_of_nodes()
			edges_count = sample_graph.number_of_edges()

			print('	nodes: {} edge: {}'.format(nodes_count, edges_count))

			#save_sample(sample_graph, output)





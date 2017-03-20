from __future__ import division, print_function
import networkx as nx
import numpy as np
import _mylib
import csv
import query
import argparse
import log
import os

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

	print(' MAX', offset)
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

def simmulate(steps):
	# Start simulating.
	for step in steps:
		nodes, edges, c = query.neighbors(str(step))
		for e in edges:
			sample_graph.add_edge(e[0], e[1])
	return sample_graph

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)

	args = parser.parse_args()
	fname = args.fname
	dataset = args.dataset

	G = _mylib.read_file(fname)

	graph = max(nx.connected_component_subgraphs(G), key=len)
	#print('LCC: # nodes', graph.number_of_nodes())
	query = query.UndirectedSingleLayer(G)

	if dataset == None:
		f = fname.split('.')[1].split('/')[2]
		dataset = f

	Log_result = {}
	fn = './log/exp1-real/' + dataset + '_order.txt'

	print(dataset, fn)

	a = read_file_step(fn)

	print(G.number_of_nodes())
	print(graph.number_of_nodes())



	to_log = []
	to_log.append(dataset)

	for id in [0,1,3,4,5]:
		#track_edges = []
		#b = []

		nodes_count_all = []
		edges_count_all = []

		for trail in range(0, 10):
			print('Running .. ', id, trail)

			steps, budget = get_line(a, set=trail, id=id)
			cost = 0
			sample_graph = nx.Graph()

			queried_nodes = (np.array(steps, dtype=str).tolist())
			sample_graph = simmulate(queried_nodes)
			#sample_graph = graph.subgraph(queried_nodes)

			nodes_count = sample_graph.number_of_nodes()
			edges_count = sample_graph.number_of_edges()

			print('	NNN {} {}'.format(nodes_count, edges_count))

			nodes_count_all.append(nodes_count)
			edges_count_all.append(edges_count)

		# for 1 algorithm
		avg_node = np.mean(np.array(nodes_count_all))
		avg_edge = np.mean(np.array(edges_count_all))

		sd_node =  np.std(np.array(nodes_count_all))
		sd_edge = np.std(np.array(nodes_count_all))

		print ('{} {} {} {}'.format(avg_node, avg_edge, sd_node, sd_edge))

		to_log.append(avg_node)
		to_log.append(sd_node)
		to_log.append(avg_edge)
		to_log.append(sd_edge)

	fn_log = './log/summary_real.csv'

	if not os.path.isfile(fn_log):
		f = open(fn_log, 'a')
		header = ['network',
				  'rw.avg.node','rw.sd.node', 'rw.avg.edge', 'rw.sd.edge',
				  'rand.avg.node', 'rand.sd.node','rand.avg.edge',  'rand.sd.edge',
				  'bfs.avg.node', 'bfs.sd.node', 'bfs.avg.edge', 'bfs.sd.edge',
				  'sb.avg.node', 'sb.sd.node', 'sb.avg.edge', 'sb.sd.edge',
				  'mod.avg.node', 'mod.sd.node', 'mod.avg.edge', 'mod.sd.edge',
				  ]
		header = str(header).replace('[', '')
		header = header.replace(']', '')

		print(header, file=f)

		to_log = str(to_log).replace('[', '')
		to_log = to_log.replace(']', '')
		print(to_log, file=f)
	else:
		f = open(fn_log, 'a')
		to_log = str(to_log).replace('[','')
		to_log = to_log.replace(']','')
		print(to_log, file=f)







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



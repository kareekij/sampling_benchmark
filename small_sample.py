from __future__ import division, print_function
import networkx as nx
import numpy as np
import _mylib
import csv
import query
import argparse
import log

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

	#print(a[2])

	print(' MAX', offset)
	start = set*offset
	end = (set*offset + offset)-1
	print('		start {} : end {}'.format(start, end))

	r = (a[id][start:end+1])

	if len(r) == 0:
		print('No data')

	return r, offset

def SaveToFile(results):
	log_file = './' + log_folder +'/' +  dataset + '_edges.txt'

	log.save_to_file(log_file, results)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)
	parser.add_argument('-type', help='sampling typess', default='bfs')
	parser.add_argument('-log_folder', help='log folder', default='./log/data3')
	parser.add_argument('-p', help='percent', default=10)


	args = parser.parse_args()
	print(args)

	fname = args.fname
	dataset = args.dataset
	type = args.type
	log_folder =  args.log_folder
	p = int(args.p)

	if dataset == None:
		f = fname.split('.')[1].split('/')[2]
		dataset = f


	G = _mylib.read_file(fname)
	print('Original: # nodes', G.number_of_nodes())
	graph = max(nx.connected_component_subgraphs(G), key=len)
	print('LCC: # nodes', graph.number_of_nodes())
	query = query.UndirectedSingleLayer(G)

	Log_result = {}

	fn = './' + log_folder +'/' + dataset + '_order.txt'
	print('Reading ', fn)
	a = read_file_step(fn)

	if type == 'rw':
		id = 0
	elif type == "random":
		id = 1
	elif type == "bfs":
		id = 3
	elif type == "sb":
		id = 4
	elif type == "mod":
		id = 5


	track_edges = []
	b = []

	for trail in range(0, 1):
		print('Running .. ', id, trail)

		steps, budget = get_line(a, set=trail, id=id)
		cost = 0
		print('	Total queried nodes', len(steps))
		init_size = int((p/100)*len(steps))
		print('	Init sample size {} nodes, sampling type: {}'.format(init_size, type))

		#small_sample_steps = steps[:init_size]
		small_sample_steps = steps
		queried_nodes = (np.array(small_sample_steps, dtype=str).tolist())

		small_sample = graph.subgraph(queried_nodes)

		nodes_count = small_sample.number_of_nodes()
		edges_count = small_sample.number_of_edges()





		# avg_deg_G = np.median(np.array(graph.degree().values()))
		# avg_deg_S = np.median(np.array( small_sample.degree().values()))
		#
		# print('Original {}, S: {} '.format(avg_deg_G, avg_deg_S))


		# small_sample_steps = steps[:init_size]
		# t = (np.array(small_sample_steps, dtype=str).tolist())
		#
		# sample_graph = nx.Graph()
		# # Start simulating.
		# for step in small_sample_steps:
		# 	nodes, edges, c = query.neighbors(str(step))
		# 	for e in edges:
		# 		sample_graph.add_edge(e[0], e[1])




		# print(sample_graph.number_of_nodes())
		# tt = sample_graph.subgraph(t)
		# print(tt.number_of_nodes())
		# print(sm.number_of_nodes())

		#avg_deg_G = np.mean(np.array(G.degree().values()))

		#avg_deg_S = np.mean(np.array( sample_graph.degree().values()))




		#print(avg_deg_G, avg_deg_S)




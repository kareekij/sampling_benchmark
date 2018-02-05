from __future__ import division, print_function
import numpy as np
import networkx as nx
import community
import _mylib
import csv
import argparse
import operator

def read_com(type):
	partition = {}
	#with open('./data/com-5000-'+type+'.dat', 'rb') as csvfile:
	with open('./data/'+mode+'/'+str(com_size)+'/community.dat', 'rb') as csvfile:
		r = csv.reader(csvfile, delimiter='\t')
		for row in r:
			partition[row[0]] = row[1]

	return partition

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)

	args = parser.parse_args()
	fname = args.fname

	f = fname.split('.')[1].split('/')[-1]
	dataset = f

	G = _mylib.read_file(fname)

	print('Original: # nodes', G.number_of_nodes())
	graph = max(nx.connected_component_subgraphs(G), key=len)
	print('LCC: # nodes', graph.number_of_nodes())

	avg_nb_deg = nx.average_neighbor_degree(graph)
	node_deg = graph.degree()

	sorted_deg = sorted(node_deg.items(), key=operator.itemgetter(1), reverse=True)
	top_k = int(.01*len(sorted_deg))

	print('Top-k ', top_k)
	d = dict()
	for item in sorted_deg[:top_k]:#:
		node = item[0]
		deg = item[1]
		print(node, deg)

		d[node] = abs(int(node_deg[node]) - int(avg_nb_deg[node]))

	print(nx.info(graph))
	mean = round(np.mean(np.array(d.values())), 2)
	med = round(np.median(np.array(d.values())),2)

	_mylib.degreeHist(node_deg.values())
	#_mylib.distributionPlot(d.values(), log_log=False, title=dataset+"Mean: "+ str(mean) +"Med: " + str(med) )




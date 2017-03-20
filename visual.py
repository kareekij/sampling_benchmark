from __future__ import division, print_function

import networkx as nx
import argparse
import _mylib
import numpy as np
import csv

def plot_cc_vs_deg(cc,deg):

	bins = {}

	for node, node_deg in deg.iteritems():
		if node_deg not in bins:
			bins[node_deg] = [cc[node]]
		else:
			bins[node_deg].append(cc[node])

	for k,v in bins.iteritems():
		bins[k] = np.mean(np.array(bins[k]))

	_mylib.scatterPlot(bins.keys(),bins.values(),save=True,xlabels='degree',ylabels='avg. clustering coeff.',title="Facebook-combined")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-l', help='largest component', type=bool, default=False)
	parser.add_argument('-directed', help='is directed graph', type=bool, default=True)


	args = parser.parse_args()
	fname = args.fname
	is_lcc = args.l
	is_directed = args.directed

	print(args)
	G = _mylib.read_file(fname)
	_mylib.draw_graph_tool_ori(G)
	#
	# if is_directed:
	# 	G = nx.read_edgelist(fname,create_using=nx.DiGraph())
	# else:
	# 	G = nx.read_edgelist(fname)
	#
	# if is_lcc:
	# 	G = max(nx.weakly_connected_component_subgraphs(G), key=len)
	# 	#G = max(nx.connected_component_subgraphs(G), key=len)
	#
	#
	#
	#
	# first_l = set()
	# with open(fname, 'rb') as csvfile:
	#
	# 	spamreader = csv.reader(csvfile, delimiter=' ')
	#
	# 	for row in spamreader:
	# 		first_l.add(row[0])
	#
	# sub_g = G.subgraph(first_l)
	# print(sub_g.number_of_nodes())
	# print(sub_g.number_of_edges())
	#
	# _mylib.draw_graph_tool_ori(sub_g)

	# _mylib.draw_graph_tool(G, list(first_l))


	# for g in sorted(nx.weakly_connected_component_subgraphs(G),reverse=True):
	# 	_mylib.draw_graph_tool(g, list(first_l))
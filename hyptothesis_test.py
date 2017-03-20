from __future__ import division, print_function
import numpy as np
import networkx as nx
import community
import _mylib
import csv
import argparse

def deg_vs_avg_nb_deg(G):
	avg_nb_deg = nx.average_neighbor_degree(G)
	deg = nx.degree(G)
	avg_deg = np.mean(np.array(deg.values()))

	x = []
	y = []
	for k, v in avg_nb_deg.iteritems():
		# print(k, v, deg[k])
		x.append(deg[k])
		y.append(v)

	_mylib.scatterPlot(x, y, save=True, xlabels="degree", ylabels="avg.neighbor degree", title=fname + ' avg.deg:' + str(round(avg_deg,2)))

def deg_vs_avg_nb_cc(G):
	deg = nx.degree(G)
	cc = nx.clustering(G)
	avg_cc = np.mean(np.array(cc.values()))

	x = []
	y = []
	for node in G.nodes():
		nbs = G.neighbors(node)
		nbs_cc = [cc.get(key) for key in nbs]
		# cc = nx.clustering(G,nbs)
		# avg_ng_cc = np.mean(np.array(cc.values()))
		avg_ng_cc = np.mean(np.array(nbs_cc))
		x.append(deg[node])
		y.append(avg_ng_cc)

	_mylib.scatterPlot(x, y, save=True, xlabels="degree", ylabels="avg.neighbor cc", title=fname + ' cc:' + str(round(avg_cc,2)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='file')

	args = parser.parse_args()
	print(args)

	fname = args.fname

	G = _mylib.read_file(fname)

	#G = nx.read_edgelist(fname, comments="%")



	#print(nx.info(G))

	#deg_vs_avg_nb_deg(G)
	#deg_vs_avg_nb_cc(G)

from __future__ import division, print_function
import _mylib
import os
import networkx as nx
import pickle

if __name__ == '__main__':
	folder = './data/twitter'

	# Read graph files from folder
	G_Layers = []
	fname = []
	for file in os.listdir(folder):
		path = folder + '/' + file
		if os.path.isfile(path):
			G = nx.Graph()
			G = _mylib.read_file(path)
			G_Layers.append(G.copy())
			fname.append(file)

	common_nodes = set(G_Layers[0].nodes())
	for i, g in enumerate(G_Layers):
		nodes = set(g.nodes())
		common_nodes = common_nodes.intersection(nodes)
		print(len(nodes), fname[i])


	# for i, g in enumerate(G_Layers):
	# 	name = fname[i].split('.')[0]
	# 	dest = folder + '/' + name + '_common.pickle'
	# 	print(' Pickle > ',dest)
	# 	sub_g = g.subgraph(list(common_nodes))
	# 	pickle.dump(sub_g, open(dest, 'wb'))

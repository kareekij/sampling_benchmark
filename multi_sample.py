from __future__ import division, print_function
import _mylib
import multi_layers as mtl
import argparse
import os
import networkx as nx


def init_multi_layer(folder):
	# Read graph files from folder
	G_Layers = []
	for file in os.listdir(folder):
		path = folder + '/' + file
		if os.path.isfile(path) and file != '.DS_Store':
			G = nx.Graph()
			G = _mylib.read_file(path)
			G_Layers.append(G.copy())
	# End Reading

	# Initialize
	multi_layers = mtl.MultiLayersNetwork()
	multi_layers.init_layer(G_Layers, folder)

	return multi_layers

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder', help='folder', default='./data/twitter-c')
	args = parser.parse_args()

	folder = args.folder
	print('> Accessing folder', folder)

	multi_layers = init_multi_layer(folder)

	exp_layer = multi_layers._layers['expensive'][0]
	chp_1_layer = multi_layers._layers['cheap'][0]
	chp_2_layer = multi_layers._layers['cheap'][1]

	print(exp_layer.number_of_nodes())
	print(chp_1_layer.number_of_nodes())
	print(chp_2_layer.number_of_nodes())



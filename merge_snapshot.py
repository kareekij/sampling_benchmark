import argparse
import os
import networkx as nx
import _mylib
import pickle

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('output', help='snapshot name')
	#parser.add_argument('sub_id', help='subreddit id')
	#parser.add_argument('sub_name', help='subreddit name')

	args = parser.parse_args()
	print(args)

	output = args.output
	#sub_id = args.sub_id
	#sub_name = args.sub_name


	path = './data/reddit-{}/'.format(output)

	compose_g = nx.Graph()
	for file in os.listdir(path):
		fname = path + file
		if os.path.isfile(fname) and file != '.DS_Store':
			G = nx.Graph()
			G = _mylib.read_file(fname)
			compose_g = nx.compose(compose_g, G, name='merge graph')


	print(compose_g.number_of_nodes())
	pickle.dump(compose_g, open(path+'merge_{}.pickle'.format(output), 'wb'))

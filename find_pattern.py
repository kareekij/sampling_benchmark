from __future__ import division, print_function
import numpy as np
import networkx as nx
import community
import _mylib
import csv

def read_com(type):
	partition = {}
	#with open('./data/com-5000-'+type+'.dat', 'rb') as csvfile:
	with open('./data/'+mode+'/'+str(com_size)+'/community.dat', 'rb') as csvfile:
		r = csv.reader(csvfile, delimiter='\t')
		for row in r:
			partition[row[0]] = row[1]

	return partition

if __name__ == '__main__':
	type = 'o'
	mode = 'com-deg'
	com_size = 10
	#G = nx.read_edgelist('./data/syn-5000-'+type+'.dat')
	#G = nx.read_edgelist('./data/'+mode+'/'+str(com_size)+'/network.dat')

	# print(G.number_of_nodes())
	# print(nx.info(G))
	# print('max deg:',max(G.degree().values()))
	# print('min deg:', min(G.degree().values()))
	# #print('cc:', nx.average_clustering(G))
	# #partition = community.best_partition(G)
	#
	# #comm = set(partition.values())
	# #print('Com found', len(comm))
	#
	# PARTITION = read_com(com_size)
	#
	# comm = set(PARTITION.values())
	# print('Q', community.modularity(PARTITION,G))
	#
	#
	#
	#
	# count_l = []
	# print('Com found', len(comm))
	# for p in comm:
	# 	members = _mylib.get_members_from_com(p,PARTITION)
	# 	count_l.append(len(members))
	#
	# print(max(count_l))
	# print(min(count_l))
	# _mylib.degreeHist(G.degree().values())

	dataset = 'syn'
	#G = nx.read_edgelist('./data/twitter2/twitter_friends.csv', delimiter=',')
	# G = nx.read_edgelist('./data/undergrad_edges')
	# G = max(nx.connected_component_subgraphs(G), key=len)
	#
	# print(nx.info(G))
	# cc = nx.average_clustering(G)
	# print(cc)
	# print('Avg Path:', nx.average_shortest_path_length(G))
	# print('-' * 10)
	#
	# G = nx.read_edgelist('./data/gen_models/'+dataset+'_rand.txt')
	# print(nx.info(G))
	# cc = nx.average_clustering(G)
	# print('CC:', cc)
	# print('Avg Path:', nx.average_shortest_path_length(G))
	# print('-' * 10)
	#
	# G = nx.read_edgelist('./data/gen_models/'+dataset+'_sw.txt')
	# print(nx.info(G))
	# cc = nx.average_clustering(G)
	# print('CC:', cc)
	# print('Avg Path:', nx.average_shortest_path_length(G))
	# print('-' * 10)
	#
	# G = nx.read_edgelist('./data/gen_models/'+dataset+'_pa.txt')
	# print(nx.info(G))
	# cc = nx.average_clustering(G)
	# print('CC:', cc)
	# print('Avg Path:', nx.average_shortest_path_length(G))
	# print('-' * 10)

	#G = nx.read_edgelist('./data/gen_models/' + dataset + '_pc.txt')
	#G = nx.read_edgelist('./data/mix-param/0.8/network.dat')
	G = nx.read_edgelist('./data/com-size/100/network.dat')
	print(nx.info(G))

	d = G.degree()
	print(min(d.values()))
	print(max(d.values()))
	cc = nx.average_clustering(G)

	print('CC:', cc)
	p = community.best_partition(G)
	print(len(set(p.values())))
	#print('Avg Path:', nx.average_shortest_path_length(G))
	#deg = G.degree()
	#_mylib.degreeHist(deg.values())
	print('-' * 10)


	# G_rand = nx.read_edgelist('./data/gen_models/random_graph_5000_0.0036.txt')
	# G_sw = nx.read_edgelist('./data/gen_models/small_world_5000_0.3.txt')
	# G_pa = nx.read_edgelist('./data/gen_models/pa_5000_9.txt')
	#
	# print(nx.info(G_rand))
	# cc = nx.average_clustering(G_rand)
	# deg = G_rand.degree()
	# print('CC: ',cc)
	# print('Avg Path:', nx.average_shortest_path_length(G_rand))
	#
	# print('-'*10)
	# print(nx.info(G_sw))
	# cc = nx.average_clustering(G_sw)
	# deg = G_sw.degree()
	# print('CC: ', cc)
	# print('Avg Path:', nx.average_shortest_path_length(G_sw))
	#
	# print('-' * 10)
	# print(nx.info(G_pa))
	# cc = nx.average_clustering(G_pa)
	# deg = G_pa.degree()
	# print('CC: ', cc)
	# print('Avg Path:', nx.average_shortest_path_length(G_pa))

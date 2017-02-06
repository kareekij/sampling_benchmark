# -*- coding: utf-8 -*-
"""
Logs data
"""
from __future__ import print_function
import time

def writeUndirectedSingleLayer(fname, dataset, budget, bfs_budget,\
 cost_expansion, cost_densification, exp_cut_off, den_cut_off, sample,\
 new_nodes, score_exp_list, score_den_list, node_observed_count):
	if dataset is None or fname is None:
		return None
	f = open(fname, 'a')
	print('='*80, file=f)
	print('Time: {}'.format(time.time()), file=f)
	# print('Dataset: {}'.format(dataset), file=f)
	print('Budget: {}'.format(budget), file=f)
	print('BFS budget: {}'.format(bfs_budget), file=f)
	# print('Oracle expansion cost: {}'.format(cost_expansion), file=f)
	# print('Oracle densification cost: {}'.format(cost_densification), file=f)
	# print('Expansion candidate count: {}'.format(exp_cut_off), file=f)
	# print('Densification candidate count: {}'.format(den_cut_off), file=f)
	# print('Number of close nodes {}'.format(len(sample['nodes']['close'])), file=f)
	# print('Number of open nodes {}'.format(len(sample['nodes']['open'])), file=f)
	# print('score_den', file=f)
	# print([round(x,4) for x in score_den_list], file=f)
	# print('score_exp',file=f)
	# print([round(x,4) for x in score_exp_list], file=f)
	# print('new_nodes',file=f)
	# print(new_nodes, file=f)
	# print('nodes observed count', file=f)
	# print(node_observed_count, file=f)
	print('nodes observed: {}'.format(len(sample['nodes']['close']) + len(sample['nodes']['open'])), file=f)
	print('='*80, file=f)
	print(' ', file=f)

def log_anything(fname,data):
	f = open(fname, 'a')
	print('=' * 80, file=f)

	for d in data:
		print([x for x in d], file=f)

	print('=' * 80, file=f)

def log_ego_graph(ego_nodes,ego_edges):
	f = open('./log/ego.txt', 'a')

	print('nodes ' + str([round(x, 4)  for x in ego_nodes]), file=f)
	print('edges ' + str([round(x, 4)  for x in ego_edges]), file=f)


# (_logfile, self._dataset, type, log_cost, sample._track_obs_nodes, self._budget, self._bfs_count)
def log_new_nodes(fname, dataset, type, new_nodes, cost_track, budget, bfs_budget):
	if dataset is None or fname is None:
		return None

	f = open(fname, 'a')

	#print('=' * 80, file=f)
	# print('Dataset: {}'.format(dataset), file=f)
	# print('Budget: {}'.format(budget), file=f)
	# print('BFS budget: {}'.format(bfs_budget), file=f)
	# print('Expansion Type: {}'.format(type), file=f)
	# print('Cost track - New nodes', file=f)
	print(type + '_Cost, ' + str([round(x, 4) for x in cost_track]), file=f)
	print(type + ', ' + str([round(x, 4) for x in new_nodes]), file=f)
	# print(type + str(k) + '_open, ' + str([round(x, 4) for x in open_l]), file=f)
	# print(type + str(k) + '_new_nodes, ' + str([round(x, 4) for x in avg_l['unobs']]), file=f)
	# print(type + str(k) + '_e_open, ' + str([round(x, 4) for x in avg_l['open']]), file=f)
	# print(type + str(k) + '_e_close, ' + str([round(x, 4) for x in avg_l['close']]), file=f)


	#print([round(x, 4) for x in obs_nodes_cheap], file=f)
	#print('=' * 80, file=f)
	#print(' ', file=f)

def write_cost_spent_return(fname, dataset, budget, bfs_budget, exp_type, cost_spent, node_return):

	if dataset is None or fname is None:
		return None
	f = open(fname, 'a')

	print('='*80, file=f)
	print('Time: {}'.format(time.time()), file=f)
	print('Dataset: {}'.format(dataset), file=f)
	print('Budget: {}'.format(budget), file=f)
	print('BFS budget: {}'.format(bfs_budget), file=f)
	print('Expansion type: {}'.format(exp_type), file=f)
	print('Cost spent, nodes return', file=f)
	# for n in cost_spent:
	# 	print(n, file=f)

	print([round(x,4) for x in cost_spent], file=f)
	print([round(x, 4) for x in node_return], file=f)


	#print('Nodes return', file=f)
	#print([round(x, 4) for x in node_return], file=f)
	print('=' * 80, file=f)

	


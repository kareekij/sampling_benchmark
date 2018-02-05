from __future__ import division, print_function
import networkx as nx
import numpy as np
import _mylib
import csv
import query
import argparse
import log
import os
import community
import pickle
from scipy import stats

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-dataset', help='Name of the dataset', default=None)

	args = parser.parse_args()
	fname = args.fname
	dataset = args.dataset

	if dataset == None:
		f = fname.split('.')[1].split('/')[-1]
		dataset = f

	G = _mylib.read_file(fname)
	deg_G = G.degree()
	avg_deg_G = np.average(np.array(deg_G.values()))
	med_deg_G = np.median(np.array(deg_G.values()))
	avg_cc_G = nx.average_clustering(G)

	sample_folder = './data-control-real/' + dataset

	log_result_file = './log/run-properties-control-real.txt'
	results = []



	if not os.path.isfile(log_result_file):
		print('Creating .. {}'.format(log_result_file))

		with open(log_result_file, 'wb') as csvfile:
			wt = csv.writer(csvfile, delimiter=' ')
			wt.writerow(['dataset','type','id','avg_deg','med_deg','avg_cc','d','p'])

	with open(log_result_file, 'a') as csvfile:
		wt = csv.writer(csvfile, delimiter=' ')
		wt.writerow([dataset, 'original', -1, avg_deg_G, med_deg_G, avg_cc_G, 0.,0.])

		#results.append([dataset, 'original', -1, avg_deg_G, med_deg_G, avg_cc_G, 0.,0.])

		for file in os.listdir(sample_folder):
			sample_G = nx.Graph()
			if file.endswith(".pickle"):
				tmp = file.split('.')[0].split('_')
				type = tmp[0]
				idx = tmp[1]

				sample_G = _mylib.read_file(sample_folder + '/' + file)
				deg_S = sample_G.degree()
				avg_deg_S = np.average(np.array(deg_S.values()))
				med_deg_S = np.median(np.array(deg_S.values()))
				avg_cc_S = nx.average_clustering(sample_G)
				D, p_value = stats.ks_2samp(deg_G.values(), deg_S.values())

				wt.writerow([dataset, type, idx, avg_deg_S, med_deg_S, avg_cc_S, D, p_value])
				#results.append([dataset, type, idx, avg_deg_S, med_deg_S, avg_cc_S, D, p_value])



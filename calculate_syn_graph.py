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

	dataset = fname

	folder_name = dataset

	# avg_deg_G = list()
	# med_deg_G = list()
	# avg_cc_G = list()
	# for i in range(1,11):
	# 	input_fname = '/Users/Katchaguy/Google Drive/datasets/syn-lfr/gen-mix-0.1/' + folder_name + '/' + str(i) + '/network.dat'
	# 	G = nx.Graph()
	# 	G = _mylib.read_file(input_fname)
	# 	deg_G = G.degree()
	# 	avg_deg_G.append(np.average(np.array(deg_G.values())))
	# 	med_deg_G.append(np.median(np.array(deg_G.values())))
	# 	avg_cc_G.append(nx.average_clustering(G))
	#
	# avg_deg_all = np.average(np.array(avg_deg_G))
	# med_deg_all = np.average(np.array(med_deg_G))
	# avg_cc_all = np.average(np.array(avg_cc_G))

	#######




	sample_folder = './data-syn-mixing/' + dataset
	log_result_file = './log/run-syn-mixing.txt'
	results = []



	if not os.path.isfile(log_result_file):
		print('Creating .. {}'.format(log_result_file))

		with open(log_result_file, 'wb') as csvfile:
			wt = csv.writer(csvfile, delimiter=' ')
			wt.writerow(['dataset','type','id','avg_deg','med_deg','avg_cc','d','p'])

	with open(log_result_file, 'a') as csvfile:
		wt = csv.writer(csvfile, delimiter=' ')
		#wt.writerow([dataset, 'original', -1, avg_deg_G, med_deg_G, avg_cc_G, 0.,0.])

		#results.append([dataset, 'original', -1, avg_deg_G, med_deg_G, avg_cc_G, 0.,0.])

		for file in os.listdir(sample_folder):
			sample_G = nx.Graph()
			if file.endswith(".pickle"):
				print('\t\t ', file)
				tmp = file.split('.')[0].split('_')
				type = tmp[0]
				idx = int(tmp[1])

				# Original
				input_fname = '/Users/Katchaguy/Google Drive/datasets/syn-lfr/gen-mix-0.1/' + folder_name + '/' + str(idx+1) + '/network.dat'
				G = nx.Graph()
				G = _mylib.read_file(input_fname)
				deg_G = G.degree()
				avg_deg_G = np.average(np.array(deg_G.values()))
				med_deg_G = np.median(np.array(deg_G.values()))
				avg_cc_G = nx.average_clustering(G)

				wt.writerow([dataset, 'original', -1, avg_deg_G, med_deg_G, avg_cc_G, 0., 0.])
				###

				sample_G = _mylib.read_file(sample_folder + '/' + file)
				deg_S = sample_G.degree()
				avg_deg_S = np.average(np.array(deg_S.values()))
				med_deg_S = np.median(np.array(deg_S.values()))
				try:
					avg_cc_S = nx.average_clustering(sample_G)
				except ZeroDivisionError:
					avg_cc_S = -1.
					continue

				D, p_value = stats.ks_2samp(deg_G.values(), deg_S.values())

				wt.writerow([dataset, type, idx, avg_deg_S, med_deg_S, avg_cc_S, D, p_value])
				#results.append([dataset, type, idx, avg_deg_S, med_deg_S, avg_cc_S, D, p_value])
	#


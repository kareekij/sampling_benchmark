from __future__ import division, print_function
import csv
import argparse
import numpy as np


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('fname', type=str, help='filename')

	args = parser.parse_args()
	fname = args.fname

	data = dict()
	#dataset	type	id	avg_deg	med_deg	avg_cc	d	p

	trial_check = dict()
	with open(fname, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		reader.next()
		for row in reader:
			print(row)
			key = row[0] + '_' + row[1]
			trial = row[2]
			avg_deg = float(row[3])
			med_deg = float(row[4])
			avg_cc = float(row[5])
			d = float(row[6])
			p = float(row[7])

			data.setdefault(key, dict())
			data[key]['avg_deg'] = data[key].get('avg_deg', list()) + [avg_deg]
			data[key]['med_deg'] = data[key].get('med_deg', list()) + [med_deg]
			data[key]['avg_cc'] = data[key].get('avg_cc', list()) + [avg_cc]
			data[key]['d'] = data[key].get('d', list()) + [d]
			data[key]['p'] = data[key].get('p', list()) + [p]

			trial_check[row[0]] = trial

	for k,v in trial_check.iteritems():
		print(k, v)

	with open('./log/summary-'+fname+'.txt', 'wb') as csvfile:
		wt = csv.writer(csvfile, delimiter=' ')

		for key in data.keys():
			print(key)
			col_1 = key.split('_')[0]
			col_2 = key.split('_')[1]
			avg_deg = np.average(np.array(data[key]['avg_deg']))
			med_deg = np.average(np.array(data[key]['med_deg']))
			avg_cc = np.average(np.array(data[key]['avg_cc']))
			d = np.average(np.array(data[key]['d']))
			p = np.average(np.array(data[key]['p']))

			wt.writerow([col_1, col_2, avg_deg, med_deg, avg_cc, d, p])

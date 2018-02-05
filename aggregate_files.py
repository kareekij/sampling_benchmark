from __future__ import division, print_function
import csv
import argparse
import numpy as np


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('fname', type=str, help='filename')

	#args = parser.parse_args()
	#fname = args.fname

	dataset_check = dict()

	with open('/Users/Katchaguy/Desktop/prop-check-result/real_properties-all.txt') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=' ')
		for row in reader:
			d = row['dataset']
			if d not in dataset_check.keys():
				dataset_check[d] = 1

	print(dataset_check.keys())
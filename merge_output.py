from __future__ import division, print_function
import csv
import argparse
import sys

def read_file(fname):
	print(' > Reading {} ...'.format(fname))

	read_data = []
	with open(fname, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			read_data.append(row)
	return read_data

def merge_files(data_1, data_2):

	row_f1 = data_1[0]
	row_f2 = data_2[0]

	print(row_f1)
	key_to_id_1 = dict()
	for idx, txt in enumerate(row_f1):
		key_to_id_1[txt.replace(' ','')] = idx

	key_to_id_2 = dict()
	for idx, txt in enumerate(row_f2):
		key_to_id_2[txt.replace(' ','')] = idx


	print(key_to_id_1)
	print(key_to_id_2)

	idx_1 = key_to_id_1['budget']
	idx_2 = key_to_id_2['budget']
	print(idx, idx_2)

	merge_data = []
	for i, row in enumerate(data_1):
		new_row = []
		if i != 0:
			budget_1 = data_1[i][idx_1]
			budget_2 = data_2[i][idx_2]

			if budget_1 != budget_2:
				print('Something wrong b1: {}, b2: {}'.format(budget_1, budget_2))
				break

			del data_2[i][idx_2]
			new_row = data_1[i] + data_2[i]
			#print(new_row)
		else:
			del data_2[i][idx_2]
			new_row = data_1[i] + data_2[i]

		merge_data.append(new_row)

		with open(OUTPUT, 'a') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',')
			spamwriter.writerow(new_row)









if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname1', help='file 1', type=str)
	#parser.add_argument('fname2', help='file 2', type=str)
	args = parser.parse_args()

	fname_1 = args.fname1
	#fname_2 = args.fname2


	for type in ['_n', '_e', '_order']:
		PATH_1 = '/Users/Katchaguy/Google Drive/results/imc2017/realworld/'
		PATH_2 = './log-large/'
		TYPE = type
		OUTPUT = './log-merge/' + fname_1 + TYPE + '.txt'

		data_1 = read_file(PATH_1 + fname_1 + TYPE + '.txt')
		data_2 = read_file(PATH_2 + fname_1 + 'exp' + TYPE + '.txt')



		if len(data_1) != len(data_2):
			print(' ** Two files do not have equal length .. {} {} '.format(len(data_1), len(data_2)))
			print(' Exit..')
			sys.exit()

		print(' > Length matches')

		merge_files(data_1, data_2)
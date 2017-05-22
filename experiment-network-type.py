import csv
import numpy as np
import argparse
import os

def process(filename):
	data = []
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			data.append(row)

	header = data[0]
	header_dict = {}

	for i, h in enumerate(header):
		h = str(h).replace(' ', '')
		header_dict[h] = i

	data_np = np.array(data)
	data_np = np.delete(data_np, (0), axis=0).tolist()

	MAX_BUDGET = data_np[-1:][0][header_dict['budget']]

	result = []
	for row in data_np:
		budget = row[header_dict['budget']]
		if budget == MAX_BUDGET:
			result.append(row)

	result = np.array(result).astype(np.int)
	result_t = (result.transpose())

	# Calculate Mean and SD
	log = {}
	for k, v in header_dict.iteritems():
		algo_name = k
		idx = v
		avg = np.mean(np.array(result_t[idx]))
		sd = np.std(result_t[idx], ddof=1)
		log[algo_name] = [avg, sd]

	return log, header_dict


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('file1', help='file 1', type=str)
	parser.add_argument('count1', help='count 1', type=int)

	#parser.add_argument('count2', help='count 2', type=int)
	#parser.add_argument('out', help='output', type=str)

	args = parser.parse_args()

	filename_1 = args.file1
	count_1 = args.count1
	print(args)
	#count_2 = args.count2


	# folder = './realworld'
	# target_type = 'n'
	# final_log = []
	# for file in os.listdir(folder):
	# 	if file.endswith(".txt"):
	# 		filepath = (os.path.join(folder, file))
	# 		type = file.split('.')[0][-1]
	# 		if type == target_type:
	#filename_1 = './realworld/ca-citeseer_n.txt'
	#count_1 = 5000
	log, header = process(filename_1)

	# Calculate Percent Improvement
	################# LOW ##########################
	results = []
	write_dict = {}
	headers = set()
	for key in log.keys():
		#if key != 'budget' and key != 'rw':

		if key == 'bfs' or key == 'mod' or key == 'rw':
			avg = 100*log[key][0] / count_1
			sd = 100*log[key][1]/count_1
			write_dict[key+'.avg'] = avg
			write_dict[key+'.sd'] = sd
			print(filename_1.split('.'))
			write_dict['network'] = filename_1.split('.')[1].split('/')[2]

			headers.add(key + '.avg')
			headers.add(key + '.sd')
			headers.add('network')
	results.append(write_dict)
	#print(write_dict)

	with open('./log/exp-network-type-edges.txt', 'a') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=list(headers))

		#writer.writeheader()
		for row in results:
			writer.writerow(row)
			print(row)

		#writer.writerows(write_dict)
			#writer.writerow(row)

	# with open('./log/test.txt', 'w') as outcsv:
	# 	writer = csv.writer(outcsv)
	# 	writer.writerow(["Level", "type", "val","sd"])
	#
	# 	for n in final_log:
	# 		writer.writerow(n)
	# ################# LOW ##########################
	#
	#
	#
	# with open(output, 'wb') as outcsv:
	# 	writer = csv.writer(outcsv)
	# 	writer.writerow(["Level", "type", "val", "sd"])
	#
	# 	for n in final_log:
	# 		print(n)
	# 		writer.writerow(n)
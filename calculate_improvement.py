import csv
import numpy as np
import argparse

def process(filename, dummy=False):
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

	MAX_BUDGET = data_np[-1:][0][header_dict['budget']].replace(' ','')
	print(MAX_BUDGET)
	if dummy:
		MAX_BUDGET = "500"#str(int(MAX_BUDGET) / 2)


	result = []
	for row in data_np:
		budget = row[header_dict['budget']].replace(' ','')
		print(budget, MAX_BUDGET)
		if budget == MAX_BUDGET:
			result.append(row)

	result = np.array(result).astype(np.int)
	result_t = (result.transpose())

	print('here')
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
	parser.add_argument('file2', help='file 2', type=str)
	parser.add_argument('count1', help='count 1', type=int)
	parser.add_argument('count2', help='count 2', type=int)
	parser.add_argument('out', help='output', type=str)

	args = parser.parse_args()

	print(args)
	filename_1 = args.file1
	filename_2 = args.file2
	output = args.out
	count_1 = args.count1
	count_2 = args.count2



	final_log = []

	log, header = process(filename_1)

	# Calculate Percent Improvement
	################# LOW ##########################
	exp_type = 'Low'
	print(log['rw'])
	for key in log.keys():
		#if key != 'budget' and key != 'rw':
		if key == 'bfs' or key == 'mod':
			avg_a = 100*log[key][0]/count_1
			avg_b = 100*log['rw'][0]/count_1
			sd_a = 100*log[key][1]/count_1
			sd_b = 100*log['rw'][1]/count_1

			# avg_imp = 100 * ((log[key][0] - log['rw'][0]) / log['rw'][0])
			# sd_imp = 100 * ((log[key][1] - log['rw'][1]) / log['rw'][1])
			avg_imp = 100 * (avg_a - avg_b) / avg_b
			sd_imp = 100 * (sd_a - sd_b) / sd_b
			#print(exp_type, avg_a, avg_b, sd_a, sd_b, '--- ', avg_imp, sd_imp)
			final_log.append([exp_type, key, avg_imp, sd_a])

	with open('./log/test.txt', 'w') as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(["Level", "type", "val","sd"])

		for n in final_log:
			writer.writerow(n)
	################# LOW ##########################

	################# HIGH ##########################
	log, header = process(filename_2)

	exp_type = 'High'
	print(log['rw'])

	for key in log.keys():
		#if key != 'budget' and key != 'rw':
		if key == 'bfs' or key == 'mod':
			avg_a = 100 * log[key][0] / count_2
			avg_b = 100 * log['rw'][0] / count_2
			sd_a = 100 * log[key][1] / count_2
			sd_b = 100 * log['rw'][1] / count_2

			avg_imp = 100 * (avg_a - avg_b) / avg_b
			#sd_imp = 100 * (sd_a - sd_b) / sd_b

			# avg_imp = 100 * ((log[key][0] - log['rw'][0]) / log['rw'][0])
			# sd_imp = 100 * ((log[key][1] - log['rw'][1]) / log['rw'][1])
			#print(exp_type, avg_a, avg_b, sd_a, sd_b, '--- ', avg_imp, sd_imp)
			final_log.append([exp_type, key, avg_imp, sd_a])

	################# HIGH ##########################

	with open(output, 'wb') as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(["Level", "type", "val", "sd"])

		for n in final_log:
			print(n)
			writer.writerow(n)
from __future__ import division, print_function
import argparse
import os
import csv

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder', help='reddit folder', default='./data/twitter')
	args = parser.parse_args()

	folder = args.folder
	print(folder)

	user2id = {}
	for file in os.listdir(folder):
		print('Reading: {}'.format(file))
		filename = file.split('.')[0]
		with open(folder+'/'+file) as csvfile:
			reader = csv.reader(csvfile, delimiter=' ')
			for row in reader:
				user_a = row[0]
				user_b = row[1]

				user2id[user_a] = user2id.get(user_a, len(user2id))
				user2id[user_b] = user2id.get(user_b, len(user2id))


				with open(folder+'/'+filename+'_edges.csv', 'a') as csvfile:
					spamwriter = csv.writer(csvfile, delimiter=' ')
					spamwriter.writerow([user2id[user_a], user2id[user_b]])

	with open(folder + '/user2id.csv', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=' ')
		for k,v in user2id.iteritems():
			spamwriter.writerow([k,v])
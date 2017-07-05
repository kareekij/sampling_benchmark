from __future__ import division, print_function
import sqlite3
import csv
import argparse
import os

def save_to_file(data, output_file):
	import csv
	with open(output_file, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=' ')
		for row in data:
			spamwriter.writerow(row)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('file', help='snapshot name')
	#parser.add_argument('sub_id', help='subreddit id')
	parser.add_argument('sub_name', help='subreddit name')

	args = parser.parse_args()
	print(args)

	#file = args.file
	#sub_id = args.sub_id
	sub_name = args.sub_name


	#path = '/Volumes/BBHDD-BL/reddit1415/'
	path = '/Users/Katchaguy/Desktop/reddit1415/'
	#path = './reddit/'
	#file = 'RC_2015-02'
	#fname = path + file + '.sqlite'


	for file in os.listdir(path):
		fname = path + file
		if os.path.isfile(fname) and file != '.DS_Store':
			conn = sqlite3.connect(fname)
			c = conn.cursor()


			SQL_STATEMENT = "SELECT id from subreddits WHERE name='{}' ".format(sub_name)
			cursor = conn.execute(SQL_STATEMENT)

			sub_id = -1

			for row in cursor:
				sub_id = row[0]

			if sub_id == -1:
				print('No {} found'.format(sub_name))
				continue


			print('Query from {} Sub name:{} id: {}'.format(fname, sub_name, sub_id))
			SQL_STATEMENT = "SELECT * from edges WHERE subreddit={} ".format(sub_id)
			cursor = conn.execute(SQL_STATEMENT)

			data = []
			for row in cursor:
				from_id = row[0]
				to_id = row[1]
				timestamp = row[2]
				comment_id = row[3]
				subreddit = row[4]

				data.append([from_id, to_id])

			directory = './data/reddit-{}/'.format(sub_name)
			if not os.path.exists(directory):
				os.makedirs(directory)

			save_to_file(data, directory + file + '_'+ sub_name +'.txt')



	# save_to_file(data,'./data/'+file+'_subreddit.txt')
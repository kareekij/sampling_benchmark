
from __future__ import division, print_function
import networkx as nx
import argparse
import random
import _mylib

class EpidemicSimmulation(object):
	def __init__(self, graph, total_nodes, sim_time, infected_init_count):
		super(EpidemicSimmulation, self).__init__()
		self._graph = graph
		self._total_nodes= total_nodes
		self._population = {'suscept': set(), 'infect': set(), 'recover': set(), 'seen':set()}
		self._time = 0
		self._sim_time = sim_time
		self._infected_init_count = infected_init_count
		self._beta = 0.3
		self._gamma = 0.1
		self._logging = dict()


	def start(self):

		self.initialize()
		nodes_seen = set()
		nodes_seen.update(self._population['infect'])


		while self._time <= self._sim_time:
			infect = self._population['infect']
			recover = self._population['recover']

			new_infected = set()
			new_recover = set()

			# Start
			for infected_node in infect:
				suscept_nbs = set(self._graph.neighbors(infected_node)) - (infect.union(recover))
				nodes_seen.update(suscept_nbs)

				# S -> I
				for s in suscept_nbs:
					r = random.uniform(0,1)

					if r <= self._beta:
						new_infected.add(s)

				# I -> R
				r = random.uniform(0,1)
				if r <= self._gamma:
					new_recover.add(infected_node)

			# End

			self._time += 1
			self.update_population(new_infected, new_recover, nodes_seen)

			total = len(self._population['suscept']) + len(self._population['infect']) + len(self._population['recover'])
			print('{} \t {} {} - S:{} , I:{} , R:{}'.format(self._time, total, self._total_nodes, len(self._population['suscept']),
												 len(self._population['infect']), len(self._population['recover'])))

	def initialize(self):
		nodes = self._graph.nodes()

		infect = set(random.sample(nodes, self._infected_init_count))
		suscept = set(nodes) -  infect

		self._population['infect'].update(infect)
		self._population['suscept'].update(suscept)

		print('Start with {} infected node(s)'.format(len(self._population['infect'])))

	def update_population(self, new_infected, new_recover, nodes_seen):

		self._population['infect'].update(new_infected)
		self._population['infect'] = self._population['infect'] - new_recover

		self._population['recover'].update(new_recover)

		self._population['suscept'] = self._population['suscept'] - new_infected

		# Logging
		self._logging['t'] = self._logging.get('t', list()) + [self._time]
		self._logging['S'] = self._logging.get('S', list()) + [len(self._population['suscept'])]
		self._logging['I'] = self._logging.get('I', list()) + [len(self._population['infect'])]
		self._logging['R'] = self._logging.get('R', list()) + [len(self._population['recover'])]
		self._logging['seen'] = self._logging.get('seen', list()) + [len(nodes_seen)]

	def log_to_file(self, filename,i):
		f = open(filename, 'a')

		if i == 0:
			print('t, s, i, r, seen', file=f)

		for idx, v in enumerate(self._logging['t']):
			t = str(self._logging['t'][idx])
			s = str(self._logging['S'][idx])
			i = str(self._logging['I'][idx])
			r = str(self._logging['R'][idx])
			se = str(self._logging['seen'][idx])
			line = t +', '+ s +', ' + i +', ' + r +', '+se
			print(line, file=f)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-simtime', help='simmulation time', type=int, default=10)
	parser.add_argument('-infected', help='# of infected node', type=int, default=1)
	parser.add_argument('-experiment', help='# of experiment', type=int, default=10)


	args = parser.parse_args()
	fname = args.fname
	sim_time = args.simtime
	infected_init_count = args.infected
	experiment = args.experiment

	f = fname.split('.')[1].split('/')[-1]
	dataset = f

	G = _mylib.read_file(fname)
	graph = max(nx.connected_component_subgraphs(G), key=len)
	total_nodes = graph.number_of_nodes()

	for i in range(0, experiment):
		simmulation = EpidemicSimmulation(graph, total_nodes, sim_time, infected_init_count)
		simmulation.start()
		simmulation.log_to_file('./log/e_'+dataset+'.txt', i)


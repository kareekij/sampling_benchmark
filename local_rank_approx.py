from __future__ import division, print_function
import networkx as nx
import argparse
import _mylib
import random
import scipy.stats as stats


def get_rank_correlation(d_1, d_2, k=.5):
	cutoff = int(k * len(d_1))
	d_1_sorted = _mylib.sortDictByValues(d_1,reverse=True)

	l_1 = []
	l_2 = []
	for count, d in enumerate(d_1_sorted):
		id = d[0]
		val = d[1]

		l_1.append(val)
		l_2.append(d_2[id])

		if count == cutoff:
			#print(count)
			break

	tau, p_value = stats.kendalltau(l_1, l_2)

	#print('correlation: {}, p-value {} \t k {}'.format(round(tau,4), round(p_value,4),k))
	print(k, tau, p_value)
	return k, tau, p_value

def log_to_file(filename,i):
	f = open(filename, 'a')

	if i == 0:
		col_names = Logging.keys()
		print(col_names, file=f)

	print(Logging)

	for idx, v in enumerate(Logging['core']):
		t = str(Logging[50][idx])
		s = str(Logging[20][idx])
		i = str(Logging[10][idx])
		r = str(Logging['core'][idx])
		se = str(Logging['ratio'][idx])
		line = t +', '+ s +', ' + i +', ' + r +', '+ str(se)
		#print(line)
		print(line, file=f)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)
	parser.add_argument('-percent', help='subgraph percent size', type=int, default=0.5)
	parser.add_argument('-core', help='# cores', type=int, default=10)

	args = parser.parse_args()

	fname = args.fname
	percent = args.percent
	core = args.core
	dataset = fname.split('.')[1].split('/')[-1]


	# Start
	Logging = dict()
	G = _mylib.read_file(fname)

	print('Original: # nodes', G.number_of_nodes())
	graph = max(nx.connected_component_subgraphs(G), key=len)
	print('LCC: # nodes', graph.number_of_nodes())

	#int((percent/100) * graph.number_of_nodes())
	for exp in range(0, 10):
		print(exp, core)
		core_nodes = random.sample(graph.nodes(), core)
		periphery_nodes = set()

		for c in core_nodes:
			nbs = set(graph.neighbors(c)) - set(core_nodes)
			periphery_nodes.update(nbs)



		all_nodes = set(core_nodes).union(set(periphery_nodes))
		sub_graph = graph.subgraph(all_nodes)
		ratio = (sub_graph.number_of_nodes()/graph.number_of_nodes())
		print('-'*10)
		print('Core: {}, Peri:{} -- Total: {}'.format(len(core_nodes), len(periphery_nodes),
													  len(core_nodes) + len(periphery_nodes)))
		print('New subgraph size', sub_graph.number_of_nodes())
		print('Original graph size', graph.number_of_nodes())
		print('Ratio', ratio )
		print('-' * 10)


		act_deg = graph.degree()
		obs_deg = sub_graph.degree()

		predicted_set = set()
		for n in sub_graph.nodes():
			d1 = act_deg[n]
			d2 = obs_deg[n]

			if d1 != d2:
				predicted_set.add(n)

		print(' > {} nodes are needed to be predicted'.format(len(predicted_set)))

		obs_deg = sub_graph.degree(predicted_set)
		act_deg = graph.degree(predicted_set)

		for k in [50,20,10]:
			k1, tau, p_value = get_rank_correlation(obs_deg, act_deg, k=50/100)
			Logging[k] = Logging.get(k, list()) + [tau]

		Logging['core'] = Logging.get('core', list()) + [core]
		Logging['ratio'] = Logging.get('ratio', list()) + [ratio]

		log_to_file('./log/rank_' + dataset + '.txt', exp)



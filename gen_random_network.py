import networkx as nx
import _mylib

G = nx.erdos_renyi_graph(n=2000, p=0.01)
density = _mylib.calculate_density(G)

print(density)

nx.write_edgelist(G, './gen-network/erdos_'+str(density))
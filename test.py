import networkx as nx
import _mylib


g = _mylib.read_file('./data/RC_2015-01.txt')
print(nx.info(g))

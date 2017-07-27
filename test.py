import networkx as nx
import _mylib
import community


g = _mylib.read_file('./data/socfb-Amherst41.mtx')
print(nx.info(g))
p = community.best_partition(g)
print(community.modularity(p,g))

for pp in set(p.values()):
	members = _mylib.get_members_from_com(pp,p)
	if len(members) > 100:
		print(pp, len(members))

print(max(p.values()))
print(min(p.values()))
print(len( set(p.values())))
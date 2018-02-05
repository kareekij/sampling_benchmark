import networkx as nx
import _mylib
import community


#g = _mylib.read_file('./data-control-real/twitter_combined.txt')
g = _mylib.read_file('./LFR-benchmark/network.dat')
density = _mylib.calculate_density(g)
print(density)
# print(nx.info(g))


# p = community.best_partition(g)
# print(community.modularity(p,g))
# #
#
# for pp in set(p.values()):
# 	members = _mylib.get_members_from_com(pp,p)
# 	print(pp, len(members))
#
# print(max(p.values()))
# print(min(p.values()))
# print(len( set(p.values())))
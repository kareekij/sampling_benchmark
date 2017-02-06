import networkx as nx
import community
import oracle
import math
import numpy as np
import matplotlib.pyplot as plt
import _mylib
import numpy as np
import re
import matplotlib.cm as cm
import exp_den as ed
import pydot
from sklearn import datasets, linear_model
from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn import metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing
import pandas as pd
import cPickle as pickle
import os
import random


#def _localClusteringCoeffDirected(node):

def countMembersInCommunity():
    partitionSet = set(o.partition.values())

    membersCount = {}
    for p in partitionSet:
        nodes = getAllNodesInCom(p)
        membersCount[p] = len(nodes)
    return membersCount

def getAllNodesInCom(p):
    valueList = np.array(o.partition.values())
    indices = np.argwhere(valueList == p)
    indices = indices.reshape(len(indices))

    keyList = np.array(o.partition.keys())

    nodes = keyList[indices]
    return nodes

def getAllNodesInIntermediateCom(p, partition):
    valueList = np.array(partition.values())
    indices = np.argwhere(valueList == p)
    indices = indices.reshape(len(indices))

    keyList = np.array(partition.keys())

    nodes = keyList[indices]
    return nodes

def _calulate_coefficient_directed(nodes_list, graph):
    ret = {}

    for node in nodes_list:
        neighbors = set(graph.neighbors(node))
        count = 0
        for neighbor in neighbors:
            n_n = set(graph.neighbors(neighbor))
            intercept = neighbors.intersection(n_n)
            count += len(intercept)
        n = len(neighbors)
        if n <= 1:
            cc = 0.
        else:
            cc = 1. * count / ((n * n) - n)

        ret[node] = cc

    return ret

def _get_all_nodes_connect_outside(nodes_list):
    ret = []
    for node in nodes_list:
        neighbors = G.neighbors(node)
        if len(set(neighbors) - set(nodes_list)) > 0:
            ret.append(node)

    print "- Connect to outside %s out of %s " % (len(ret),len(nodes_list))
    return ret

def _density(com_id):
    nodesInCom = getAllNodesInCom(com_id)
    sub_g = G.subgraph(nodesInCom).to_undirected()
    #sub_g = G.subgraph(nodesInCom)

    edges_count = sub_g.number_of_edges()
    nodes_count = sub_g.number_of_nodes()

    t = (nodes_count - 1)
    n_c_r = (nodes_count * (nodes_count - 1)) / 2.
    #n_c_r = (nodes_count * (nodes_count - 1))

    try:
        density = 1.*(edges_count - t) / (n_c_r - t)
    except ZeroDivisionError:
        density = 0.

    print 'density: com', com_id, edges_count, n_c_r, density

    return density

def _density_from_sub_graph(sub_graph):
    e = sub_graph.number_of_edges()
    n = sub_graph.number_of_nodes()
    d = 2. * e / (n * (n - 1))
    return d

def readText():
    f = open('./data/open-nodes.txt', 'r')
    txt = f.read()#.split(',')
    open_nodes =  re.findall(r"'(.*?)'", txt, re.DOTALL)


    f = open('./data/close-nodes.txt', 'r')
    txt = f.read()  # .split(',')
    close_nodes = re.findall(r"'(.*?)'", txt, re.DOTALL)

    #print len(open_nodes), len(close_nodes), (len(open_nodes) + len(close_nodes))

    all_nodes = set(open_nodes).union(set(close_nodes))
    print len(all_nodes)

    check_com = {}
    for node in close_nodes:
        p =  o.partition[node]
        check_com[p] = check_com.get(p,0) + 1


    sub_g = G.subgraph(all_nodes)

    print nx.info(sub_g)

    for k, v in check_com.iteritems():
        print k, v, membersCount[k]

    print 'open node'
    check_com = {}

    for node in open_nodes:
        p = o.partition[node]
        check_com[p] = check_com.get(p, 0) + 1

    for k, v in check_com.iteritems():
        print k, v, membersCount[k]

def avgCC_degreeHist():
    cc_nodes = _calulate_coefficient_directed(G.nodes(), G)
    out_deg_nodes = G.out_degree(G.nodes())
    in_deg_nodes = G.in_degree(G.nodes())

    t_1 = {}
    t_2 = {}
    for node, degree in out_deg_nodes.iteritems():
        cc = cc_nodes[node]
        degree = degree
        tmp = t_1.get(degree, [])
        tmp.append(cc)
        t_1[degree] = tmp
        # print len( t_1[degree])

    avg_list = []
    std_list = []
    var_list = []

    for k, v in t_1.iteritems():
        avg = np.average(t_1[k])
        std = np.std(t_1[k])
        var = np.var(t_1[k])
        # print avg, std
        avg_list.append(avg)
        std_list.append(std)
        var_list.append(var)

        # t_1[degree] = t_1.get(degree,0) + cc
        # t_2[degree] = t_2.get(degree,0) + 1

    # degree_hist =  np.array(t_1.values()) / np.array(t_2.values())r
    # print std_list[:50]

    print std_list[:50]

    fig, ax = plt.subplots()
    MIN = 0
    MAX = len(t_1.keys())
    rects1 = ax.bar(t_1.keys()[MIN:MAX], avg_list[MIN:MAX], color='r', yerr=var_list[MIN:MAX])

    plt.show()

def FindY():
    y = {}

    for node in G.nodes():
        node_com =  com = o.partition[node]
        neighbors = G.neighbors(node)

        count = 0
        for n in neighbors:
            com = o.partition[n]
            if com != node_com:
                count+=1

        try:
            score = 1.*count / len(neighbors)
        except ZeroDivisionError:
            score = 0.

        y[node] = score

    return y

def print_info(G):
    print '------- Graph --------'
    print nx.info(G)
    print 'isDirected:', nx.is_directed(G)
    print 'Mod %s' % community.modularity(o.partition,G)
    cc = nx.clustering(G, G.nodes())
    print 'Avg. CC %s ' % np.average(cc.values())
    print '----------------------'

def Features_Extraction():
    com_fname = 'data/features_{}.pickle'.format(dataset)

    if os.path.isfile(com_fname):
        print 'Get features from pickle file.. '
        features = pickle.load(open(com_fname, 'rb'))
    else:
        print 'Extracting features.. '
        degree, degree_avg_nb, cc, cc_avg_nb, ego_edges, ego_edges_out = degree_cc(G)
        deg_cen = nx.degree_centrality(G)
        page_rank = nx.pagerank(G)
        #betweeness = nx.betweenness_centrality(G)

        # Put all features to a single dictionary, key is a nodeID
        features = {}
        for key in G.nodes():
            feature = []
            feature.append(degree[key])
            feature.append(degree_avg_nb[key])
            feature.append(cc[key])
            feature.append(cc_avg_nb[key])
            feature.append(ego_edges[key])
            feature.append(ego_edges_out[key])
            feature.append(deg_cen[key])
            feature.append(page_rank[key])
            #feature.append(betweeness[key])

            features[(key)] = feature

        pickle.dump(features, open(com_fname, 'wb'))

    return features

def degree_cc(G):
    degree = G.degree(G.nodes())
    cc = _calulate_coefficient_directed(G.nodes(), G)

    degree_avg_nb = {}
    cc_avg_nb = {}
    ego_edges = {}
    ego_edges_out = {}

    for node in G.nodes():
        nbs = G.neighbors(node)
        ego_G = nx.ego_graph(G,node)
        ego_edges[node] = ego_G.number_of_edges()
        ego_edges_out[node] = ego_net_edges_out(ego_G)


        sum_nb_deg = 0
        sum_nb_cc = 0
        for nb in nbs:
            sum_nb_deg += degree[nb]
            sum_nb_cc += cc[nb]
        degree_avg_nb[node] = 1.*sum_nb_deg / len(nbs)
        cc_avg_nb[node] = 1.*sum_nb_cc / len(nbs)

    return degree, degree_avg_nb, cc, cc_avg_nb, ego_edges, ego_edges_out

def ego_net_edges_out(ego_net):
    count = 0
    for node in ego_net.nodes():
        deg_ego = ego_net.degree(node)
        deg_G = G.degree(node)
        count += (deg_G - deg_ego)
    return count

def classify_inner_border():
    features = Features_Extraction()
    features_keys = np.array(features.keys())
    features_matrix = np.matrix(features.values())

    #min_max_scaler = preprocessing.MinMaxScaler()
    #features_matrix = min_max_scaler.fit_transform(features_matrix)

    #print features_matrix_norm[0]

    print "Features Matrix shape", features_matrix.shape

    com_graph = {}
    y_label = {}

    for node in features.keys():
        com = o.partition[node]

        if com_graph.get(com, 0) == 0:
            members = getAllNodesInCom(com)
            com_graph[com] = G.subgraph(members)

        c = com_graph[com]

        node_deg_in_G = G.neighbors(node)
        node_deg_in_S = c.neighbors(node)

        if len(node_deg_in_G) - len(node_deg_in_S) == 0:
            y_label[node] = 0  # inner
        else:
            y_label[node] = 1  # boarder

    print 'Running classification algorithm..'
    COUNT = 10

    y = np.array(y_label.values())
    sss = StratifiedShuffleSplit(y, COUNT, test_size=0.7, random_state=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    i = 1
    log = []
    for train_index, test_index in sss:
        # nodes = np.concatenate([ getAllNodesInCom(1), getAllNodesInCom(2)])
        #
        # ix = np.in1d(features_keys.ravel(), nodes).reshape(features_keys.shape)
        # train_index = np.where(ix)[0]
        #
        # nodes_test = list( set(G.nodes()).difference(set(nodes)) )
        # ix = np.in1d(features_keys.ravel(), nodes_test).reshape(features_keys.shape)
        # test_index = np.where(ix)[0]

        X_train, X_test = features_matrix[train_index], features_matrix[test_index]
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)

        Y_train, Y_test = y[train_index], y[test_index]

        # Split the data into training/testing sets
        # X_train = features_matrix[:-30]
        # X_test = features_matrix[-30:]


        # Split the targets into training/testing sets
        # Y_train = y[:-30]
        # Y_test = y[-30:]
        print " ------ %s ------ " % i
        print "Training  ", len(X_train)
        print "Testing   ", len(Y_test)

        # Decition Tree
        #clf = tree.DecisionTreeClassifier()
        #clf = clf.fit(X_train, Y_train)
        #Y_predict = clf.predict(X_test)

        # SVM
        #clf = svm.SVC(kernel='linear', C=1.).fit(X_train, Y_train)
        #Y_predict = clf.predict(X_test)

        # kNN
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_train, Y_train)
        Y_predict = neigh.predict(X_test)
        #Y_predict = neigh.predict_proba(X_test)

        fpr, tpr, thresholds = metrics.roc_curve(Y_test.tolist(), Y_predict.tolist())
        acc = accuracy_score(Y_test, Y_predict)
        auc_score = metrics.auc(fpr, tpr)
        auc_score2 = roc_auc_score(Y_test,Y_predict)
        f1 = f1_score(Y_test, Y_predict)

        print "Accuracy", acc
        print "AUC score", auc_score, auc_score2
        print "F1 score", f1
        print "---------------------------------"

        log.append([i,acc,auc_score, f1])
        i += 1

    _mylib.logToFileCSV(log,filename="./classification/"+dataset+"_knn5.csv")


        # plt.figure(1)
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr, tpr)
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title('ROC curve')
        # plt.legend(loc='best')
        # plt.show()

def get_sub_features_mat(features, nodes):
    features_matrix = np.matrix(features.values())
    features_keys = np.array(features.keys())

    ix = np.in1d(features_keys.ravel(), nodes).reshape(features_keys.shape)

    nodes_indices = np.where(ix)[0]

    sub_ = features_matrix[nodes_indices]
    return sub_

def get_close_open_node(node_type,type=0):
    # type 0: close otherwise opened node
    x = np.array(node_type.values())
    ix = np.in1d(x.ravel(), [type]).reshape(x.shape)
    nodes_indices = np.where(ix)[0]

    return np.array(node_type.keys())[nodes_indices]


def features_correlation_check():
    features = Features_Extraction()
    nodes_border = []
    nodes_inner = []
    #for com_id in membersCount.keys():
    for com_id in range(3,4):
        nodes_in_com = getAllNodesInCom(com_id)
        com_G = G.subgraph(nodes_in_com)
        cc_com_G = nx.clustering(com_G, nodes_in_com)

        print "Community %s , %s nodes " % (com_id, len(nodes_in_com))
        for node in nodes_in_com:
            deg_G = G.degree(node)
            deg_C = com_G.degree(node)

            if deg_G > deg_C:
                nodes_border.append(node)

            else:
                nodes_inner.append(node)

    print "Total nodes %s b: %s i: %s " % (G.number_of_nodes(), len(nodes_border), len(nodes_inner))

    sub_mat_inner = get_sub_features_mat(features, nodes_inner)
    sub_mat_border = get_sub_features_mat(features, nodes_border)

    print '-- Inner -- '
    _mylib.pairwise_correlation_matrix(sub_mat_inner)
    print '-- Border -- '
    _mylib.pairwise_correlation_matrix(sub_mat_border)

def BFS(count, graph=None,start_node=None):
    if graph == None:
        graph = G

    if start_node == None:
        nodes = graph.nodes()
        rand = random.randint(0, len(nodes) - 1)
        start_node = nodes[rand]

    sample_edges = set()
    sample_nodes_core = set()
    sample_nodes_peri = set()

    visited, queue = set(), [start_node]
    index = 0
    node_type = {}
    while index < count :
        if len(queue) == 0:
            print "Empty Queue, run again"
            sys.exit()
            break
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            neighbors = set(graph.neighbors(vertex))
            node_type[vertex] = 0 # 0 = closed node
            for nb in neighbors:
                a = vertex
                b = nb
                if node_type.get(nb,-1) == -1:
                    node_type[nb] = 1 # 1 = opened node
                # if vertex < nb:
                #     a = vertex
                #     b = nb
                # elif nb < vertex:
                #     a = nb
                #     b = vertex

                sample_edges.add((a,b))

            queue.extend(neighbors - visited)
            index +=1

    sample_nodes_core = visited
    sample_nodes_peri = set(queue).difference(set(visited))


    return sample_edges, sample_nodes_core, sample_nodes_peri, node_type

def sub_com_score(a,b,node_type,closed_nodes, opened_nodes):
    a = set(a)
    b = set(b)
    score_1 = 1.*len((a.intersection(b))) / len(b) # how many nodes in b are in original
    score_2 = 1.*len((a.intersection(b))) / len(a) # how many nodes in a that b can capture

    type_count = {}
    for node in b:
        type = node_type[node]
        #print ' %s %s ' % (node, type)
        type_count[type] = type_count.get(type,0) + 1

    score_3 = 1.*type_count.get(0,0) / len(b) # score of closed node over all nodes in B
    score_4 = 1.*type_count.get(0,0) / len(closed_nodes)


    return score_1, score_2,score_3,score_4

def count_members_com(sample_g,partition):
    check_comp = {}
    for node in sample_g.nodes():
        p = partition[node]
        check_comp[p] = check_comp.get(p, 0) + 1

    print '------- count members -------'
    print _mylib.sortDictByValues(check_comp,reverse=True)
    print '-----------------------------'

def sub_graph_cd(com_id=2):
    # Get all nodes in community
    nodes_in_com = getAllNodesInCom(com_id)
    actual_com = G.subgraph(nodes_in_com)
    print "[GT] Actual density", _density_from_sub_graph(actual_com)

    # Random starting node from com_id and create ground-truth community (sub_g)
    sub_g_original = G.subgraph(nodes_in_com)
    rand = random.randint(0, len(nodes_in_com) - 1)
    start_node = nodes_in_com[rand]

    # From a selected node, perform BFS and get sub sample
    print "Start at node", start_node
    edges, closed_nodes, opened_nodes, node_type = BFS(40, G, start_node)
    print "Closed %s , Opened %s " % (len(closed_nodes), len(opened_nodes))

    print len(get_close_open_node(node_type, 0))
    print len(get_close_open_node(node_type, 1))

    # Create sub sample graph from BFS and perform community detection
    sample_G = nx.Graph()
    sample_G.add_edges_from(edges)
    partition = o.partition
    partition_i = community.best_partition(sample_G)

    count_members_com(sample_G,partition_i)

    print "Sub sample graph Mod: %s" % (community.modularity(partition_i, sample_G))

    p_i = list(set(partition_i.values()))

    print "Density before %s " % _density_from_sub_graph(sample_G)

    members_com = {}
    density = {}
    scores = {}

    # Check score for each new community in sub sample graph
    print "    s1     s2    s3      s4   den     size   d+size"
    for p in p_i:
        members = getAllNodesInIntermediateCom(p, partition_i)
        s1, s2, s3, s4 = sub_com_score(nodes_in_com, members, node_type, closed_nodes, opened_nodes)
        s = sample_G.subgraph(members)
        d = _density_from_sub_graph(s)

        com_size = 1. * len(members) / (sample_G.number_of_nodes())

        members_com[p] = members
        density[p] = d * com_size  # + s4
        scores[p] = [s1, s2, s3, s4, com_size]

        print "%s:| %.2f - %.2f | %.2f - %.2f | %.2f | %.2f | %.2f" % (p, s1, s2, s3, s4, d, com_size, density[p])
        # count_members_com(s)

    print '--------------------------'

    # Pick a community with the highest score.
    density_sorted = _mylib.sortDictByValues(density, reverse=True)
    cur_density = 0.
    # new_density = 0.
    cur_members = []
    MIN_DEN = 0.2
    count_members = 0
    partition_tmp = {}
    for d in density_sorted:
        sub_com_id = d[0]
        # density = d[1]
        members = members_com[sub_com_id].tolist() + cur_members
        # Construct new sub graph of merged communities
        s = sample_G.subgraph(members)
        d = _density_from_sub_graph(s)
        cc = nx.number_connected_components(s)
        com_size = 1. * len(members_com[sub_com_id].tolist()) / sub_g_original.number_of_nodes()
        # print "%s: ** d=%.2f | members: %s | c-b: %.2f | c-all: %.2f" % (sub_com_id, d, len(members_com[sub_com_id].tolist()),scores[sub_com_id][2],scores[sub_com_id][3])
        print "%s: %.2f %s %.2f %.2f %.2f %.2f" % (
            sub_com_id, d, com_size, scores[sub_com_id][0], scores[sub_com_id][1], scores[sub_com_id][2],
            scores[sub_com_id][3])

        for node in sample_G.nodes():
            if node in closed_nodes:
                partition_tmp[node] = 0
            else:
                partition_tmp[node] = 10
        break

    _mylib.draw_com(sample_G, partition_tmp)

    # partition_tmp = {}
    #
    # for node in G.nodes():
    #     if node in nodes_in_com:
    #         partition_tmp[node] = 0
    #     else:
    #         partition_tmp[node] = 10
    #


if __name__ == "__main__":
    #dataset = 'Wiki-Vote'
    #dataset = 'enron'
    #dataset = 'dblp'
    #dataset = 'fb'
    dataset = 'grad'

    print 'Dataset', dataset
    if dataset == 'CA-GrQc':
        G = nx.read_edgelist('./data/CA-GrQc.txt')
        o = oracle.Oracle(G, 'CA-GrQc')
    elif dataset == 'dblp':
        G = nx.read_edgelist('./data/com-dblp.ungraph.txt')
        o = oracle.Oracle(G, 'com-dblp.ungraph')
    elif dataset == 'Wiki-Vote':
        #G = nx.read_edgelist('./data/Wiki-Vote.txt', create_using=nx.DiGraph())
        G = nx.read_edgelist('./data/Wiki-Vote.txt')
        o = oracle.Oracle(G, 'Wiki-Vote')
    elif dataset == 'grad':
        G_t = nx.read_edgelist('./data/grad_edges')
        G = max(nx.connected_component_subgraphs(G_t), key=len)
        o = oracle.Oracle(G, 'grad')
    elif dataset == 'undergrad':
        G = nx.read_edgelist('./data/undergrad_edges')
        o = oracle.Oracle(G, 'undergrad')
    elif dataset == 'enron':
        G = nx.read_edgelist('./data/Email-Enron.txt')
        o = oracle.Oracle(G, 'enron')
    elif dataset == 'fb':
        G = nx.read_edgelist('./data/107.edges')
        o = oracle.Oracle(G, 'fb')

    print_info(G)

    membersCount = countMembersInCommunity()
    print membersCount

    sub_graph_cd(0)

    #####################


    #     if d >= cur_density and cc == 1:
    #         count_members += len(members_com[sub_com_id])
    #         cur_density = d
    #         cur_members = members
    #         print ' Merge %s -- new d %.2f' % (sub_com_id,cur_density)
    #
    # print cur_members
    # s1, s2, s3 = sub_com_score(nodes_in_com, cur_members)
    # print "%.2f %.2f %.2f %s" % (s1,s2, s3, len(cur_members))
    # s = sample_G.subgraph(cur_members)
    # count_members_com(s)
    # print count_members

    # p_check = {}
    # for node in members_com[d_best_p]:
    #     p = partition[node]
    #     p_check[p] = p_check.get(p,0) + 1
    # print p_check

    #e_node = o.expansion(list(opened_nodes),sample_G.nodes())

    #print e_node[0], partition[e_node[0]]



    #print membersCount
    #classify_inner_border()



    # edges, closed_nodes, opened_nodes = ed.BFS(5, G)
    #
    # print "Core: %s - Peri: %s - Edges: %s " % (len(closed_nodes), len(opened_nodes), len(edges))
    #
    # sub_graph = nx.Graph()
    # sub_graph.add_edges_from(edges)
    # print nx.info(sub_graph)
    # print nx.is_directed(sub_graph)




    # pos = nx.random_layout(sub_graph)
    # nx.draw_networkx_nodes(sub_graph, pos,
    #                        nodelist=opened_nodes,
    #                        node_color='b',
    #                        node_size=500,
    #                        alpha=0.8)
    # nx.draw_networkx_nodes(sub_graph, pos,
    #                        nodelist=closed_nodes,
    #                        node_color='y',
    #                        node_size=500,
    #                        alpha=0.8)
    #
    # nx.draw_networkx_edges(sub_graph, pos, width=1.0, alpha=0.5)
    # plt.show()
    #

    #nx.draw_random(sub_graph)


    #plt.show()




    #print membersCount
    #print _mylib.sortDictByValues(membersCount)


    #features_correlation_check()
    #classify_inner_border()





    #cc = nx.clustering(G, G.nodes())
    #betweeness = nx.betweenness_centrality(G)

    #_mylib.correlationXY(cc.values(),betweeness.values(),x_name="CC",y_name="Betweeness")

    #
    #for com_id in range(0,com_end):



        #_mylib.correlationXY(nodes_border_deg, nodes_border_cc, save=True)
    # End Border-inner nodes, corelation



    # max_member = max(membersCount.values())
    # min_member = min(membersCount.values())
    # print "Max Min ", max_member, min_member
    # small = []
    # med = []
    # large = []
    #
    #
    # #_mylib.degreePlotHist(1,1,[membersCount.values()])
    #
    # for k,v in membersCount.iteritems():
    #     scale_value = (v - min_member) / (max_member - min_member)
    #     if v <= 500:
    #         small.append(k)
    #     elif 500 < v <= (1000):
    #         med.append(k)
    #     else:
    #         large.append(k)
    #     # if scale_value <= 0.33: small.append(k)
    #     # elif 0.33 < scale_value <= (0.66): med.append(k)
    #     # else: large.append(k)
    #
    #
    # print "Mod: %s " % (community.modularity(o.partition,G))
    # print "S: %s M:%s L:%s " % (len(small), len(med), len(large))
    #
    # cc_all = nx.clustering(G,G.nodes())
    # #deg_all = G.degree(G.nodes())
    #
    # print "Avg. CC %s " % (np.average(cc_all.values()))
    #_mylib.scatterPlot(deg_all.values(), cc_all.values(),xlabels="Degree",ylabels="CC",title="CC vs. Deg (%s) " % (dataset) )



    # Start
    # data = []
    # for index, check_list in enumerate([small,med,large]):
    #     size = 'small'
    #     if index == 0: size = 'small'
    #     elif index == 1: size = 'med'
    #     elif index == 2: size = 'large'
    #
    #     #check_list = small
    #     deg_inside_all = []
    #     deg_boarder_all = []
    #
    #     cc_inside_all = []
    #     cc_boarder_all = []
    #     print '---------- %s ----------- ' , size
    #
    #     for com_id in check_list:
    #         node_in_com = getAllNodesInCom(com_id)
    #
    #         com_graph = G.subgraph(node_in_com)
    #
    #         cc_G = nx.clustering(G, node_in_com)
    #         cc_C = nx.clustering(com_graph, node_in_com)
    #
    #         inside_nodes = []
    #         boarder_nodes = []
    #
    #
    #         #print 'Commmunity %s: %s members' % (com_id, membersCount[com_id])
    #
    #         for node in node_in_com:
    #             node_deg_in_G = G.neighbors(node)
    #             node_deg_in_S = com_graph.neighbors(node)
    #
    #             if len(node_deg_in_G) - len(node_deg_in_S) == 0:
    #                 inside_nodes.append(node)
    #             else:
    #                 boarder_nodes.append(node)
    #
    #         #print "Com %s: Boarders %s - Inside %s (%s) " % (com_id, len(boarder_nodes), len(inside_nodes), 1.*len(boarder_nodes)/len(inside_nodes))
    #
    #         degree_inside_nodes = G.degree(inside_nodes)
    #         degree_boarder_nodes = G.degree(boarder_nodes)
    #
    #         cc_inside_nodes = nx.clustering(G,inside_nodes)
    #         cc_boarder_nodes = nx.clustering(G, boarder_nodes)
    #
    #         deg_inside_all = deg_inside_all + degree_inside_nodes.values()
    #         deg_boarder_all += degree_boarder_nodes.values()
    #
    #         cc_inside_all += cc_inside_nodes.values()
    #         cc_boarder_all += cc_boarder_nodes.values()
    #
    #         row = []
    #         row.append(com_id)
    #         row.append(size)
    #         row.append(len(boarder_nodes))
    #         row.append(len(inside_nodes))
    #         row.append(1.*len(boarder_nodes)/len(inside_nodes))
    #         data.append(row)
    #
    # _mylib.logToFileCSV(data,filename='size.csv')
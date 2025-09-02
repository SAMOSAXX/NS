import random
import collections
import numpy as np
import networkx as nx
from pylab import rcParams
import matplotlib.pyplot as plt

graph1 = nx.expected_degree_graph(nx.utils.random_sequence.powerlaw_sequence(1000, exponent=2.5))
graph1 = nx.Graph(graph1)
plt.figure(figsize=(20,20))
print(nx.draw_networkx(graph1, node_color = 'g',with_labels = False, node_size=25))
plt.axis('Off')
graph2 = nx.expected_degree_graph(nx.utils.random_sequence.powerlaw_sequence(1000, exponent=2.2))
graph2 = nx.Graph(graph2)
plt.figure(figsize=(20,20))
nx.draw_networkx(graph2, node_color = 'b',with_labels = False, node_size=25)
plt.axis('Off')

graph3 = nx.expected_degree_graph(nx.utils.random_sequence.powerlaw_sequence(1000, exponent=2.2))
Graph3 = nx.Graph(graph3)
plt.figure(figsize=(20,20))
nx.draw_networkx(graph3, node_color = 'y',with_labels = False, node_size=25)
plt.axis('Off')

rm_graph1 = graph1.copy()
rm_graph1.remove_edges_from(list(rm_graph1.edges())[:1500])

rm_graph2 = graph2.copy()
rm_graph2.remove_edges_from(list(rm_graph2.edges())[:1500])

rm_graph3 = graph3.copy()
rm_graph3.remove_edges_from(list(rm_graph3.edges())[:1500])

largest_cc_rm_g1   =  max(nx.connected_components(rm_graph1), key=len)
largest_cc_g1      =  max(nx.connected_components(graph1), key=len)

largest_cc_rm_g2   =  max(nx.connected_components(rm_graph2), key=len)
largest_cc_g2      =  max(nx.connected_components(graph2), key=len)

largest_cc_rm_g3   =  max(nx.connected_components(rm_graph3), key=len)
largest_cc_g3      =  max(nx.connected_components(graph3), key=len)
print("Fraction of nodes in giant component removing 1550 edges from,")
print("Graph 1 :",len(largest_cc_rm_g1)/len(largest_cc_g1))
print("Graph 2 :",len(largest_cc_rm_g2)/len(largest_cc_g2))
print("Graph 3 :",len(largest_cc_rm_g3)/len(largest_cc_g3))

graph  = nx.expected_degree_graph(nx.utils.random_sequence.powerlaw_sequence(1000, exponent=2.2))
graph  = nx.Graph(graph2)
plt.figure(figsize=(20,20))

nx.draw_networkx(graph1, node_color = 'orange',with_labels = False, node_size=25)
plt.axis('Off')

sizeof_giant_components = []
fraction = []
for i in range(1, 1000):
    temp = graph.copy()
    temp.remove_nodes_from(list(temp.nodes())[:i])
    sizeof_giant_components.append(len(max(nx.connected_components(temp), key=len)))
    fraction.append(i)

plt.figure(figsize=(10,10))
plt.plot(np.array(fraction)/1000, np.array(sizeof_giant_components)/1000)
plt.xlabel('Fraction of nodes in giant component')


plt.ylabel('Fraction of nodes removed Fc')
plt.title('Critical Threshold')
plt.show()

graph  = nx.expected_degree_graph(nx.utils.random_sequence.powerlaw_sequence(1000, exponent=2.2))
graph  = nx.Graph(graph)
plt.figure(figsize=(20,20))
nx.draw_networkx(graph, node_color = 'pink',with_labels = False, node_size=25)
plt.axis('Off')

rm_graph    = graph.copy()
degree_list = dict(graph.degree())

rm_graph.remove_nodes_from(list(dict(sorted(degree_list.items(), key=lambda x: x[1], reverse=True)[:50]).keys()))

giant_cc_rm_graph = max(nx.connected_components(rm_graph), key=len)
giant_cc_graph    = max(nx.connected_components(graph), key=len)

print(len(giant_cc_rm_graph)/len(giant_cc_graph))
rm_graph = Graph.copy()
d = dict(nx.clustering(rm_graph))
rm_graph.remove_nodes_from(list(dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:50]).keys()))

giant_cc_rm_graph  = max(nx.connected_components(rm_graph), key=len)
giant_cc_graph     = max(nx.connected_components(graph), key=len)

print(len(giant_cc_rm_graph)/len(giant_cc_graph))

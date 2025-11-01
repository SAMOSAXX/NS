
#graph creation
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

n=1000
p_er= 0.006
m_ba = 3
k_ws= 6
p_ws = 0.1

ws_graph = nx.watts_strogatz_graph(n,k_ws,p_ws)
er_graph = nx.erdos_renyi_graph(n,p_er)
ba_graph = nx.barabasi_albert_graph(n,m_ba)

#Metrics
def metrics(G):
   clustering = nx.average_clustering(G)
   degree_ass = nx.degree_assortativity_coefficient(G)
   betweenness_centrality = nx.betweenness_centrality(G)
   diameter = nx.diameter(G)
   top_hubs = sorted(betweenness_centrality,key=betweenness_centrality.get,reverse=True)
   if nx.is_directed(G):
        G_cc = max(nx.strongly_connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
   else:
        G_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
   return clustering,degree_ass,diameter,top_hubs[:5]

metrics(ba_graph)

def sir_simulation(G,beta = 0.8,gamma =0.05,steps=100):
 status={n:"S" for n in G}
 status[0]="I"
 infected_count = []
 for _ in range(steps):
  new_status = status.copy()
  for n in list(G.nodes()):
   if status[n]=="I":
    for neighbour in G.neighbors(n):
     if status[neighbour]=="S" and np.random.rand()<beta:
      new_status[neighbour]="I"
  for n in list(G.nodes()):
            if status[n] == "I" and np.random.rand() < gamma:
                new_status[n] = "R"
    
  
  status= new_status
  current_infected = sum(1 for s in status.values() if s == "I")
  infected_count.append(current_infected)
  print(_," ",current_infected)
 
#plotting graph
def plot_graph(G):
 l = nx.spring_layout(G)
 nx.draw(G,l)

plot_graph(ba_graph)

def find_kappa(degrees):
 avg = np.mean(degrees)
 avg_sqaure = np.mean(avg**2)
 return avg/avg_sqaure

degrees = [d for _,d in ba_graph.degree()]
find_kappa(degrees)

degrees = [d for _ , d in G.degree()]
fit = Fit(degrees)
alpha = fit.power_law.alpha
xmin = fit.power_law.xmin
exp = fit.exponential.parameter1

R,p = fit.distribution_compare("power_law","exponential")
print(R,p)

def percolation_simulation(G,removal = "random",steps=100):
 N = len(G.nodes())
 components = []
 
 fraction = np.linspace(0,1,steps)
 for f in fraction:
   G_copy = G.copy()
   if removal=="targeted":
    sorted_nodes = sorted(G_copy.degree,key = lambda x:x[1],reverse=True)
    nodes_to_remove = [node[0] for node in sorted_nodes[:int(f*N)]]
   else:
    nodes_to_remove = list(G_copy.nodes())
    np.random.shuffle(nodes_to_remove)
    nodes_to_remove = nodes_to_remove[:int(f*N)]
   G_copy.remove_nodes_from(nodes_to_remove)
   if len(G_copy)==0:
    max_size=0
   else:
    comp = nx.connected_components(G_copy)
    max_size = max(len(c) for c in comp)
   components.append(max_size/N)
   #print(components)
 return fraction,components



f,c=percolation_simulation(er_graph,"targeted",100)
plt.plot(f,c)
print(c)

def critical_threshold(c,f,threshold):
 for  _, size in enumerate(c):
  print(size)
  if size<=threshold:
   return f[_]
  return 1.0
 
critical_threshold(c,f,0.1)

#comparision of Distribution
degrees = [d for _,d in er_graph.degree()]
deg_mean = np.mean(degrees)
deg_std = np.std(degrees)
N = len(er_graph)
norm = np.random.normal(deg_mean,deg_std)
poi = np.random.poisson(deg_mean,N)
uni = np.random.uniform(deg_mean-deg_std,deg_mean+deg_std,N)
plt.figure(figsize=(6,6))
plt.hist(degrees,bins=30,color="blue",alpha=0.6)
plt.hist(poi,bins=30,color="green",alpha = 0.6)
plt.hist(norm, bins=30, density = True, log = True, color = "pink", alpha=0.6)
plt.hist(uni, bins=30, density = True, log = True, color = "purple", alpha=0.6)
plt.show()
plt.show()

#Sand pile model
threshold = 5
G = nx.Graph()
p=0.3
nodes = np.arange(50)
G.add_nodes_from(nodes)

for i in range(len(nodes)):
 G.nodes[i]["label"]=0
 for j in range(i+1,len(nodes)):
  if np.random.random()<p:
   G.add_edge(i,j)


steps = 100
for _ in range(steps):
 node = np.random.randint(0,len(nodes))
 if G.nodes[node]["label"]<threshold:
  G.nodes[node]["label"]+=1
 else:
  G.nodes[node]["label"]=0
  #topple
  for nei in G.neighbours(node):
     G.nodes[nei]["label"]+=1
  

#Cascading Failure Model
er_graph = nx.erdos_renyi_graph(100,0.4)
time_steps = 50
degrees = [ d for _,d in er_graph.degree()]
connected_components =[]
G_copy = er_graph.copy()

for time in range(time_steps):
 if len(G_copy)==0:
  break
 node_to_remove = max(G_copy.degree(), key=lambda x: x[1])[0]
 G_copy.remove_node(node_to_remove)
 G_cc = nx.connected_components(G_copy)
 l_cc = max(len(g) for g in G_cc)
 connected_components.append((time,(l_cc)))
 
times, sizes = zip(*connected_components)   # Unpack list of tuples

plt.figure(figsize=(10, 6))
plt.plot(times, sizes, 'o-', color='red', linewidth=2, markersize=4)
plt.xlabel("Step (Node Removed)")
plt.ylabel("Size of Largest Connected Component")
plt.title("Cascading Failure: Targeted Attack (Highest Degree)")

#all distances
def all_distances(G, source, algorithm='bfs'):
    if algorithm == "bfs":
        path_result = list(nx.bfs_tree(G, source=source).nodes())
    elif algorithm == "dfs":
        path_result = list(nx.dfs_preorder_nodes(G, source=source))
    else:
        raise ValueError("algorithm must be 'bfs' or 'dfs'")

    edges_in_path = list(zip(path_result, path_result[1:]))

    pos = nx.spring_layout(G)

    plt.figure(figsize=(6, 4))
    nx.draw_networkx_nodes(G, pos, nodelist=path_result,
                           node_size=40, node_color='steelblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', alpha=0.3)
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_path,
                           edge_color='red', width=2)
    plt.title("Traversal")
    plt.axis('off')
    plt.show()

    return path_result

#BA N/W
n=15
m= 3
core = 5
G = nx.complete_graph(core)

for new in range(core, n):
    G.add_node(new)
    probs = np.array([G.degree(i) for i in range(new)]) 
    probs = probs / probs.sum() if probs.sum() > 0 else None
    targets = np.random.choice(range(new), size=m, replace=False, p=probs)
    for t in targets:
        G.add_edge(new, t)

plot_graph(G)

#configuration model
N = 100
Y= 2.5
kmin = 1
kmax = int(N**(1/(Y-1)))
k= np.arange(kmin,kmax+1)
probs = k**(-Y)
probs = probs/probs.sum()
degrees = np.random.choice(k,size=N,p=probs)
if sum(degrees)%2!=0:
 degrees[0]+=1


G_config = nx.configuration_model(degrees,create_using=nx.Graph)
G_config = nx.Graph(G_config)
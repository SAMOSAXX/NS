#1 the General Working and Syntaxes for BA network and its metrics

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit

# Generate Barabási-Albert network
n = 1000  # Number of nodes
m = 3     # Number of edges to attach from a new node to existing nodes
G = nx.barabasi_albert_graph(n, m)

# Basic network metrics
avg_path_length = nx.average_shortest_path_length(G)
avg_clustering = nx.average_clustering(G)
diameter = nx.diameter(G)
assortativity = nx.degree_assortativity_coefficient(G)

# Degree statistics
degrees = [d for n, d in G.degree()]
min_degree = min(degrees)
max_degree = max(degrees)
avg_degree = np.mean(degrees)

# Power-law fit for degree distribution
fit = Fit(degrees, verbose=False)
power_law_alpha = fit.power_law.alpha
power_law_xmin = fit.power_law.xmin

# Degree distribution histogram
hist, bin_edges = np.histogram(degrees, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
valid = hist > 0  # Filter for log-log plot

# Centrality measures
betweenness = nx.betweenness_centrality(G)
top_hubs = sorted(betweenness, key=betweenness.get, reverse=True)[:5]

# Percolation analysis (robustness to targeted node removal)
def percolation(G, fraction, targeted=False):
    G_copy = G.copy()
    if targeted:
        nodes_to_remove = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
    else:
        nodes_to_remove = list(G_copy.nodes())
        np.random.shuffle(nodes_to_remove)
    num_remove = int(fraction * len(G_copy))
    G_copy.remove_nodes_from(nodes_to_remove[:num_remove])
    components = list(nx.connected_components(G_copy))
    return len(max(components, key=len, default=set()))  # Size of largest component

largest_component_random = percolation(G, 0.2, targeted=False)
largest_component_targeted = percolation(G, 0.2, targeted=True)

# SIR simulation
def sir_simulation(G, beta=0.1, gamma=0.05, steps=100):
    status = {n: 'S' for n in G}
    status[0] = 'I'  # Start with node 0 infected
    infected_count = []
    for _ in range(steps):
        new_status = status.copy()
        for n in G:
            if status[n] == 'I':
                for neighbor in G.neighbors(n):
                    if status[neighbor] == 'S' and np.random.rand() < beta:
                        new_status[neighbor] = 'I'
                if np.random.rand() < gamma:
                    new_status[n] = 'R'
        status = new_status
        infected_count.append(sum(1 for s in status.values() if s == 'I'))
    return max(infected_count)

max_infected = sir_simulation(G)

# Community detection
from networkx.algorithms.community import greedy_modularity_communities
communities = greedy_modularity_communities(G)
num_communities = len(communities)
modularity = nx.community.modularity(G, communities)

# Print results
print("=== Barabási-Albert Network Analysis ===")
print(f"Average Shortest Path Length: {avg_path_length:.3f}")
print(f"Average Clustering Coefficient: {avg_clustering:.3f}")
print(f"Diameter: {diameter}")
print(f"Degree Assortativity Coefficient: {assortativity:.3f}")
print(f"Minimum Degree: {min_degree}")
print(f"Maximum Degree: {max_degree}")
print(f"Average Degree: {avg_degree:.3f}")
print(f"Power-law Exponent: {power_law_alpha:.3f}")
print(f"Power-law xmin: {power_law_xmin}")
print(f"Top 5 Hubs by Betweenness: {top_hubs}")
print(f"Largest Component after 20% Random Removal: {largest_component_random}")
print(f"Largest Component after 20% Targeted Removal: {largest_component_targeted}")
print(f"Max Infected in SIR: {max_infected}")
print(f"Number of Communities: {num_communities}")
print(f"Modularity: {modularity:.3f}")

# Plot degree distribution
plt.figure(figsize=(8, 6))
plt.loglog(bin_centers[valid], hist[valid], 'b.', label='Degree distribution')
plt.xlabel('Degree (k)')
plt.ylabel('P(k)')
plt.title('Degree Distribution of Barabási-Albert Network')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Plot network visualization
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
node_sizes = [d * 10 for n, d in G.degree()]  # Scale for visibility
nx.draw(G, pos, node_size=node_sizes, node_color='skyblue', edge_color='gray', alpha=0.6)
plt.title('Barabási-Albert Network Visualization')
plt.show()
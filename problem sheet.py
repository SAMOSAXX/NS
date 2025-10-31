#PROGRAMMING AASIGNMENT 1

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Task 1: Generate and visualize G(N, p) networks
def generate_and_visualize_erdos_renyi(N, avg_degree, title):
    p = avg_degree / (N - 1)
    G = nx.erdos_renyi_graph(N, p)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=20, node_color='blue', with_labels=False)
    plt.title(title)
    plt.show()  # Display the plot

# Task 2: Red and Blue network
def generate_red_blue_network(N, p, q):
    G = nx.Graph()
    nodes = list(range(2 * N))
    G.add_nodes_from(nodes)
    colors = {i: 'red' if i < N else 'blue' for i in nodes}
    nx.set_node_attributes(G, colors, 'color')
    for i in range(2 * N):
        for j in range(i + 1, 2 * N):
            if colors[i] == colors[j] and np.random.random() < p:
                G.add_edge(i, j)
            elif colors[i] != colors[j] and np.random.random() < q:
                G.add_edge(i, j)
    return G

def check_connectivity(G):
    return nx.number_connected_components(G)

def average_shortest_path_length(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    return float('inf')

# Task 3: Red, Blue, Purple network
def generate_red_blue_purple_network(N, f, p):
    total_nodes = 2 * N
    num_purple = int(f * total_nodes)
    num_red_blue = total_nodes - num_purple
    N_red = num_red_blue // 2
    N_blue = num_red_blue - N_red
    G = nx.Graph()
    nodes = list(range(total_nodes))
    G.add_nodes_from(nodes)
    colors = {i: 'red' if i < N_red else 'blue' if i < N_red + N_blue else 'purple' for i in nodes}
    nx.set_node_attributes(G, colors, 'color')
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if colors[i] == colors[j] and colors[i] in ['red', 'blue'] and np.random.random() < p:
                G.add_edge(i, j)
            elif colors[i] == 'purple' and colors[j] in ['red', 'blue'] and np.random.random() < p:
                G.add_edge(i, j)
            elif colors[j] == 'purple' and colors[i] in ['red', 'blue'] and np.random.random() < p:
                G.add_edge(i, j)
    return G

def check_two_step_connectivity(G, N):
    red_nodes = [n for n, attr in G.nodes(data=True) if attr['color'] == 'red']
    blue_nodes = [n for n, attr in G.nodes(data=True) if attr['color'] == 'blue']
    for red in red_nodes[:10]:  # Sample 10 nodes
        for blue in blue_nodes[:10]:
            paths = list(nx.all_simple_paths(G, red, blue, cutoff=2))
            if not any(len(path) == 3 for path in paths):
                return False
    return True

# Main execution
if __name__ == "__main__":
    N = 500

    # Task 1: Generate and display three networks
    avg_degrees = [0.8, 1, 8]
    for k in avg_degrees:
        title = f'Erdős-Rényi Network (N={N}, <k>={k})'
        generate_and_visualize_erdos_renyi(N, k, title)

    # Task 2b: Check connectivity
    p = 1 / (N - 1)
    q = 1 / N
    G_rb = generate_red_blue_network(N, p, q)
    num_components = check_connectivity(G_rb)
    print(f"Task 2b: Number of components with p={p:.6f}, q={q:.6f}: {num_components}")

    # Task 2c: Small-world property
    p_snobbish = 0.01
    q_snobbish = 0.0001
    G_snobbish = generate_red_blue_network(N, p_snobbish, q_snobbish)
    if nx.is_connected(G_snobbish):
        avg_path = average_shortest_path_length(G_snobbish)
        print(f"Task 2c: Average shortest path length (p={p_snobbish}, q={q_snobbish}): {avg_path:.2f}")
    else:
        print("Task 2c: Network is not connected")

    # Task 3a: Test two-step connectivity
    p = 8 / (N - 1)
    f_values = [0.01, 0.05, 0.1]
    for f in f_values:
        G_rbp = generate_red_blue_purple_network(N, f, p)
        is_interactive = check_two_step_connectivity(G_rbp, N)
        print(f"Task 3a: f={f}, Interactive: {is_interactive}")
        
        
        
#PROGRAMMING AASIGNMENT 2 
# BA model
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

# Task 1: Power-law degree distribution networks
def generate_power_law_sequence(N, gamma):
    # Generate power-law degree sequence
    degrees = np.random.pareto(gamma - 1, N).astype(int) + 1  # Ensure minimum degree 1
    # Adjust to make sum even
    if sum(degrees) % 2 != 0:
        degrees[np.random.randint(0, N)] += 1
    return degrees

def compute_multi_self_loops(N, gamma, trials=5):
    self_loop_percent = []
    multi_link_percent = []
    for _ in range(trials):
        # Generate configuration model
        degrees = generate_power_law_sequence(N, gamma)
        G_multi = nx.configuration_model(degrees)
        total_edges = G_multi.number_of_edges()
        
        # Count self-loops
        self_loops = sum(1 for u, v in G_multi.edges() if u == v)
        self_loop_percent.append((self_loops / total_edges) * 100 if total_edges > 0 else 0)
        
        # Convert to simple graph to count multi-links
        G_simple = nx.Graph(G_multi)
        multi_links = total_edges - G_simple.number_of_edges() - self_loops
        multi_link_percent.append((multi_links / total_edges) * 100 if total_edges > 0 else 0)
    
    return np.mean(self_loop_percent), np.mean(multi_link_percent)

def plot_percentages(N_values, gamma_values):
    for gamma in gamma_values:
        self_loops = []
        multi_links = []
        for N in N_values:
            sl, ml = compute_multi_self_loops(N, gamma)
            self_loops.append(sl)
            multi_links.append(ml)
        
        plt.figure(figsize=(8, 6))
        plt.plot(N_values, self_loops, label='Self-loops', marker='o')
        plt.plot(N_values, multi_links, label='Multi-links', marker='s')
        plt.xscale('log')
        plt.xlabel('Number of Nodes (N)')
        plt.ylabel('Percentage (%)')
        plt.title(f'Multi-links and Self-loops for γ = {gamma}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()

# Task 2: Barabási-Albert network
def generate_barabasi_albert(N, m):
    # Start with a fully connected graph of m nodes
    G = nx.complete_graph(m)
    degrees = [m-1] * m  # Initial degrees
    snapshots = []
    target_N = [100, 1000, 10000]
    
    for new_node in range(m, N):
        # Preferential attachment: choose m nodes
        probs = np.array(degrees) / sum(degrees)
        targets = np.random.choice(list(G.nodes()), size=m, replace=False, p=probs)
        G.add_node(new_node)
        for target in targets:
            G.add_edge(new_node, target)
            degrees[target] += 1
        degrees.append(m)
        
        # Snapshot at target sizes
        if new_node + 1 in target_N:
            snapshots.append(G.copy())
    
    return snapshots

def fit_power_law(degree_counts, k_min=1):
    degrees = []
    for k, count in degree_counts.items():
        degrees.extend([k] * count)
    degrees = np.array(degrees)
    degrees = degrees[degrees >= k_min]  # Filter small degrees
    if len(degrees) == 0:
        return None, None
    log_degrees = np.log(degrees)
    gamma = 1 + len(degrees) / np.sum(log_degrees - np.log(k_min))
    return gamma, len(degrees)

def plot_degree_distributions(snapshots, target_N):
    plt.figure(figsize=(8, 6))
    for i, G in enumerate(snapshots):
        degree_counts = Counter(dict(G.degree()).values())
        degrees = np.array(list(degree_counts.keys()))
        counts = np.array(list(degree_counts.values()))
        probs = counts / sum(counts)
        plt.loglog(degrees, probs, 'o', label=f'N={target_N[i]}')
        
        # Fit power-law
        gamma, n = fit_power_law(degree_counts)
        if gamma:
            k = np.array(list(degree_counts.keys()))
            fitted = k ** (-gamma) / np.sum(k ** (-gamma))
            plt.loglog(k, fitted, '--', label=f'Fit γ={gamma:.2f}')
    
    plt.xlabel('Degree (k)')
    plt.ylabel('P(k)')
    plt.title('Degree Distributions')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_cumulative_distributions(snapshots, target_N):
    plt.figure(figsize=(8, 6))
    for i, G in enumerate(snapshots):
        degree_counts = Counter(dict(G.degree()).values())
        degrees = np.array(list(degree_counts.keys()))
        counts = np.array(list(degree_counts.values()))
        sorted_degrees = np.sort(degrees)
        cum_probs = np.cumsum(counts[np.argsort(degrees)][::-1]) / sum(counts)
        plt.loglog(sorted_degrees, cum_probs, 'o-', label=f'N={target_N[i]}')
    
    plt.xlabel('Degree (k)')
    plt.ylabel('P(K ≥ k)')
    plt.title('Cumulative Degree Distributions')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

def compute_clustering_coefficient(N_values, m):
    clustering = []
    for N in N_values:
        G = nx.barabasi_albert_graph(N, m)
        clustering.append(nx.average_clustering(G))
    
    plt.figure(figsize=(8, 6))
    plt.plot(N_values, clustering, marker='o')
    plt.xscale('log')
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average Clustering Coefficient')
    plt.title('Clustering Coefficient vs N (m=4)')
    plt.grid(True, which="both", ls="--")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Task 1: Power-law networks
    N_values = [100, 500, 1000, 5000, 10000, 50000, 100000]
    gamma_values = [2.2, 3.0]
    for gamma in gamma_values:
        print(f"\nTask 1: γ = {gamma}")
        for N in [1000, 10000, 100000]:
            sl, ml = compute_multi_self_loops(N, gamma)
            print(f"N={N}: Self-loops = {sl:.2f}%, Multi-links = {ml:.2f}%")
    plot_percentages(N_values, gamma_values)

    # Task 2: Barabási-Albert network
    N = 10000
    m = 4
    target_N = [100, 1000, 10000]
    snapshots = generate_barabasi_albert(N, m)
    
    # Task 2a, 2b: Degree distributions and power-law fit
    plot_degree_distributions(snapshots, target_N)
    
    # Task 2c: Cumulative degree distributions
    plot_cumulative_distributions(snapshots, target_N)
    
    # Task 2d: Clustering coefficient
    N_values = [100, 200, 500, 1000, 2000, 5000, 10000]
    compute_clustering_coefficient(N_values, m)
    
    
#PROGRAMMING ASSIGNMENT 3
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Generate power-law degree sequence
def generate_power_law_sequence(N, gamma):
    degrees = np.random.pareto(gamma - 1, N).astype(int) + 1  # Minimum degree 1
    if sum(degrees) % 2 != 0:
        degrees[np.random.randint(0, N)] += 1
    return degrees

# Simulate attack and track giant component size
def simulate_attack(G, criterion, fractions):
    G_copy = G.copy()
    N = G_copy.number_of_nodes()
    sizes = []
    
    if criterion == 'degree':
        nodes_sorted = sorted(G_copy.nodes(), key=lambda x: G_copy.degree(x), reverse=True)
    else:  # clustering
        clustering = nx.clustering(G_copy)
        nodes_sorted = sorted(G_copy.nodes(), key=lambda x: clustering[x], reverse=True)
    
    for f in fractions:
        num_remove = int(f * N)
        G_temp = G_copy.copy()
        G_temp.remove_nodes_from(nodes_sorted[:num_remove])
        if G_temp.number_of_nodes() == 0:
            sizes.append(0)
        else:
            largest_cc = max(nx.connected_components(G_temp), key=len, default=set())
            sizes.append(len(largest_cc) / N)
    
    return sizes

# Plot giant component sizes
def plot_giant_component(fractions, sizes_degree, sizes_clustering, title):
    plt.figure(figsize=(8, 6))
    plt.plot(fractions, sizes_degree, label='Degree Attack', marker='o')
    plt.plot(fractions, sizes_clustering, label='Clustering Attack', marker='s')
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Giant Component Size (Normalized)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    N = 10000
    fractions = np.linspace(0, 0.5, 11)  # 0 to 50% in 5% steps

    # Task 1: Configuration Model (Power-law, γ = 2.5)
    degrees = generate_power_law_sequence(N, gamma=2.5)
    G_config = nx.configuration_model(degrees)
    G_config = nx.Graph(G_config)  # Convert to simple graph
    sizes_degree_config = simulate_attack(G_config, 'degree', fractions)
    sizes_clustering_config = simulate_attack(G_config, 'clustering', fractions)
    print("Configuration Model (γ = 2.5):")
    print(f"Degree Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_degree_config]}")
    print(f"Clustering Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_clustering_config]}")
    plot_giant_component(fractions, sizes_degree_config, sizes_clustering_config, 
                        'Giant Component Size vs Fraction Removed (Configuration Model, γ=2.5)')

    # Task 2: Hierarchical Model
    G_hierarchical = nx.powerlaw_cluster_graph(N, m=4, p=0.1)
    sizes_degree_hier = simulate_attack(G_hierarchical, 'degree', fractions)
    sizes_clustering_hier = simulate_attack(G_hierarchical, 'clustering', fractions)
    print("\nHierarchical Model (m=4, p=0.1):")
    print(f"Degree Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_degree_hier]}")
    print(f"Clustering Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_clustering_hier]}")
    plot_giant_component(fractions, sizes_degree_hier, sizes_clustering_hier, 
                        'Giant Component Size vs Fraction Removed (Hierarchical Model)')
    
    
    
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_config = 10000  # Number of nodes for configuration model
gamma = 2.5  # Power-law exponent
k_min = 3  # Minimum degree
steps = 20  # Number of fractions for removal simulation

# Function to generate power-law degree sequence for configuration model
def generate_powerlaw_degrees(N, gamma, k_min):
    max_degree = int(N ** (1 / (gamma - 1)))  # Finite-size cutoff
    k = np.arange(k_min, max_degree + 1)
    probs = k ** (-gamma)
    probs /= probs.sum()
    degrees = np.random.choice(k, size=N, p=probs)
    if sum(degrees) % 2 != 0:
        degrees[-1] += 1
    return degrees

# Function to generate hierarchical network (Ravasz-Barabási model, ADVANCED TOPIC 9.B)
def generate_hierarchical(level):
    if level == 1:
        G = nx.complete_graph(5)
        central = 0
        external = [1, 2, 3, 4]
        return G, central, external

    G_prev, central_prev, external_prev = generate_hierarchical(level - 1)
    G = G_prev.copy()
    node_offset = len(G)
    replicas_offsets = []
    new_external = []
    for _ in range(4):
        H = G_prev.copy()
        mapping = {u: u + node_offset for u in H.nodes()}
        H = nx.relabel_nodes(H, mapping)
        G = nx.union(G, H)
        replicas_offsets.append(node_offset)
        rep_external = [e + node_offset for e in external_prev]
        new_external += rep_external
        for ex in rep_external:
            G.add_edge(ex, central_prev)
        node_offset += len(G_prev)
    new_central = central_prev
    return G, new_central, new_external

# Function to simulate targeted removal
def simulate_targeted_removal(G, metric_key, steps=20):
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []
    # Compute metrics
    if metric_key == 'degree':
        metrics = {n: d for n, d in G.degree()}
    elif metric_key == 'clustering':
        metrics = nx.clustering(G)
    # Sort nodes by metric descending
    sorted_nodes = sorted(G.nodes(), key=lambda n: metrics[n], reverse=True)
    for f in fractions:
        num_remove = int(f * N)
        remove_list = sorted_nodes[:num_remove]
        G_copy = G.copy()
        G_copy.remove_nodes_from(remove_list)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = list(nx.connected_components(G_copy))
            largest_components.append(max(len(c) for c in components) / N)
    return fractions, largest_components

# Generate configuration model network
config_degrees = generate_powerlaw_degrees(N_config, gamma, k_min)
G_config = nx.configuration_model(config_degrees, create_using=nx.Graph)
G_config = nx.Graph(G_config)  # Remove multi-edges/self-loops

# Generate hierarchical network (level 5 for N=3125 ≈10^4; level 6=15625 is slow)
G_hier, _, _ = generate_hierarchical(5)  # N=3125

# Simulate attacks on configuration model
config_fracs_deg, config_sizes_deg = simulate_targeted_removal(G_config, 'degree', steps)
config_fracs_clus, config_sizes_clus = simulate_targeted_removal(G_config, 'clustering', steps)

# Simulate attacks on hierarchical network
hier_fracs_deg, hier_sizes_deg = simulate_targeted_removal(G_hier, 'degree', steps)
hier_fracs_clus, hier_sizes_clus = simulate_targeted_removal(G_hier, 'clustering', steps)

# Plot results for configuration model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(config_fracs_deg, config_sizes_deg, 'r-', label='High Degree Removal')
plt.plot(config_fracs_clus, config_sizes_clus, 'b-', label='High Clustering Removal')
plt.xlabel('Fraction Removed (f)')
plt.ylabel('Giant Component Size S(f)/N')
plt.title('Configuration Model (γ=2.5, N=10000)')
plt.legend()
plt.grid(True)

# Plot results for hierarchical model
plt.subplot(1, 2, 2)
plt.plot(hier_fracs_deg, hier_sizes_deg, 'r-', label='High Degree Removal')
plt.plot(hier_fracs_clus, hier_sizes_clus, 'b-', label='High Clustering Removal')
plt.xlabel('Fraction Removed (f)')
plt.ylabel('Giant Component Size S(f)/N')
plt.title('Hierarchical Model (N=3125)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
    
    
#sample code with GML file

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Function to compute network metrics
def compute_metrics(G, name):
    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    
    # Average shortest path length (for the largest connected component)
    if nx.is_directed(G):
        G_cc = max(nx.strongly_connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    else:
        G_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    avg_path_length = nx.average_shortest_path_length(G_sub)
    
    # Average degree
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    
    print(f"\nMetrics for {name}:")
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
    print(f"Average Shortest Path Length: {avg_path_length:.4f}")
    print(f"Average Degree: {avg_degree:.4f}")
    return avg_clustering, avg_path_length, avg_degree, degrees

# Function to visualize network and degree distribution
def visualize_network(G, name, degrees):
    # Create figure with two subplots: network layout and degree distribution
    plt.figure(figsize=(12, 5))
    
    # Network layout (spring layout)
    plt.subplot(121)
    pos = nx.spring_layout(G, seed=42)  # Consistent layout for reproducibility
    nx.draw(G, pos, node_size=50, node_color='skyblue', edge_color='gray', with_labels=False)
    plt.title(f"{name} Layout")
    
    # Degree distribution histogram
    plt.subplot(122)
    plt.hist(degrees, bins=20, density=True, color='salmon', edgecolor='black')
    plt.title(f"{name} Degree Distribution")
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.show()

# Load the dataset from a GML file
try:
    G_data = nx.read_gml('network.gml')
    print("Loaded network from GML file")
    avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G_data, "Input Network")
    visualize_network(G_data, "Input Network", degrees_data)
except FileNotFoundError:
    print("GML file not found. Please provide a valid GML file path.")
    # Fallback: Create a sample graph for demonstration
    G_data = nx.karate_club_graph()
    print("Using Karate Club graph as fallback")
    avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G_data, "Karate Club Network")
    visualize_network(G_data, "Karate Club Network", degrees_data)

# Number of nodes for generated networks
n = G_data.number_of_nodes()
# Approximate number of edges for realistic comparisons
m = G_data.number_of_edges()
# Estimate edge probability for Erdős-Rényi
p = (2 * m) / (n * (n - 1)) if not nx.is_directed(G_data) else m / (n * (n - 1))

# 1. Erdős-Rényi Graph
G_er = nx.erdos_renyi_graph(n, p)
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G_er, "Erdős-Rényi Graph")
visualize_network(G_er, "Erdős-Rényi Graph", degrees_er)

# 2. Watts-Strogatz Model
k = int(np.mean([d for n, d in G_data.degree()]))
G_ws = nx.watts_strogatz_graph(n, k, 0.1)
avg_clust_ws, avg_path_ws, avg_deg_ws, degrees_ws = compute_metrics(G_ws, "Watts-Strogatz Graph")
visualize_network(G_ws, "Watts-Strogatz Graph", degrees_ws)

# 3. Scale-Free Network (Barabási-Albert model)
m_ba = max(1, int(m / n)) # Ensure at least 1 edge
G_sf = nx.barabasi_albert_graph(n, m_ba)
avg_clust_sf, avg_path_sf, avg_deg_sf, degrees_sf = compute_metrics(G_sf, "Scale-Free Network")
visualize_network(G_sf, "Scale-Free Network", degrees_sf)

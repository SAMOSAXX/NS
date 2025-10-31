#Undirected

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit
import random

# ===================== Utility Functions =====================
def compute_metrics_and_distribution(G, label="Network"):
    """Compute network metrics, degree distribution, and power-law fit."""
    metrics = {}
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
    else:
        metrics['avg_path_length'] = float('inf')
        metrics['diameter'] = float('inf')

    metrics['avg_clustering'] = nx.average_clustering(G)
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)

    degrees = [d for _, d in G.degree()]
    metrics['min_degree'] = min(degrees) if degrees else 0
    metrics['max_degree'] = max(degrees) if degrees else 0
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0

    # Degree distribution
    hist, bin_edges = np.histogram(degrees, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = hist > 0

    # Power-law fit
    power_law_stats = {}
    fit = None
    if degrees:
        fit = Fit(degrees, verbose=False)
        power_law_stats['alpha'] = fit.power_law.alpha
        power_law_stats['xmin'] = fit.power_law.xmin
        R, p = fit.distribution_compare('power_law', 'exponential')
        power_law_stats['vs_exponential_R'] = R
        power_law_stats['vs_exponential_p'] = p
    else:
        power_law_stats['alpha'] = power_law_stats['xmin'] = 0
        power_law_stats['vs_exponential_R'] = 0
        power_law_stats['vs_exponential_p'] = 1

    return metrics, (hist[valid], bin_centers[valid], fit, power_law_stats)

def remove_nodes(G, fraction, removal_type='random'):
    """Remove fraction of nodes randomly or targeted (by degree)."""
    G_copy = G.copy()
    N = len(G_copy)
    num_remove = int(fraction * N)
    if removal_type == 'targeted':
        nodes_to_remove = sorted(G_copy.degree, key=lambda x: x[1], reverse=True)
        nodes_to_remove = [node[0] for node in nodes_to_remove[:num_remove]]
    else:
        nodes_to_remove = list(G_copy.nodes())
        random.shuffle(nodes_to_remove)
        nodes_to_remove = nodes_to_remove[:num_remove]
    G_copy.remove_nodes_from(nodes_to_remove)
    return G_copy

def percolation_simulation(G, removal_type='random', steps=50):
    """Simulate percolation and return largest component fraction at each step."""
    N = len(G)
    fractions = np.linspace(0, 1, steps)
    largest_components = []

    for f in fractions:
        G_copy = remove_nodes(G, f, removal_type)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            components = nx.connected_components(G_copy)
            largest = max(len(c) for c in components) / N
            largest_components.append(largest)

    return fractions, largest_components

def find_failure_threshold(fractions, largest_components, threshold=0.01):
    """Estimate percolation threshold where largest component <= threshold."""
    for i, size in enumerate(largest_components):
        if size <= threshold:
            return fractions[i]
    return 1.0

def plot_degree_distribution(hist_bins_list, labels, title="Degree Distribution"):
    plt.figure(figsize=(10,5))
    for (hist, bins), label in zip(hist_bins_list, labels):
        plt.loglog(bins, hist, marker='o', linestyle='', label=label)
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def generate_networks(n=1000, p_er=0.006, m_ba=3, k_ws=6, p_ws=0.1):
    """Generate ER, BA, and WS networks."""
    ER = nx.erdos_renyi_graph(n, p_er)
    BA = nx.barabasi_albert_graph(n, m_ba)
    WS = nx.watts_strogatz_graph(n, k_ws, p_ws)
    return ER, BA, WS

# ===================== Main Analysis =====================
if __name__ == "__main__":
    print("=== Complex Network Robustness and Scale-Free Analysis ===")
    n = 1000
    m_ba = 3
    p_er = 2*m_ba/(n-1)  # ER probability to match BA average degree ~6
    fraction_remove = 0.2  # 20% nodes removed

    # Generate networks
    ER, BA, WS = generate_networks(n=n, p_er=p_er, m_ba=m_ba)
    print("Networks generated: ER, BA, WS")

    # Compute metrics before removal
    ba_metrics, ba_data = compute_metrics_and_distribution(BA, "BA Original")
    er_metrics, er_data = compute_metrics_and_distribution(ER, "ER Original")

    # Node removal: Random and Targeted
    BA_rand = remove_nodes(BA, fraction_remove, 'random')
    BA_targ = remove_nodes(BA, fraction_remove, 'targeted')
    ER_rand = remove_nodes(ER, fraction_remove, 'random')
    ER_targ = remove_nodes(ER, fraction_remove, 'targeted')

    # Compute metrics after removal
    ba_rand_metrics, ba_rand_data = compute_metrics_and_distribution(BA_rand, "BA Random Removal")
    ba_targ_metrics, ba_targ_data = compute_metrics_and_distribution(BA_targ, "BA Targeted Removal")
    er_rand_metrics, er_rand_data = compute_metrics_and_distribution(ER_rand, "ER Random Removal")
    er_targ_metrics, er_targ_data = compute_metrics_and_distribution(ER_targ, "ER Targeted Removal")

    # Print summary metrics
    print("\nBA Original Metrics:", ba_metrics)
    print("BA 20% Random Removal Metrics:", ba_rand_metrics)
    print("BA 20% Targeted Removal Metrics:", ba_targ_metrics)
    print("\nER Original Metrics:", er_metrics)
    print("ER 20% Random Removal Metrics:", er_rand_metrics)
    print("ER 20% Targeted Removal Metrics:", er_targ_metrics)

    # Percolation simulations
    nets = {'ER': ER, 'BA': BA, 'WS': WS}
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    for name, G_net in nets.items():
        fr_rand, lc_rand = percolation_simulation(G_net, 'random')
        fr_targ, lc_targ = percolation_simulation(G_net, 'targeted')
        fc_rand = find_failure_threshold(fr_rand, lc_rand)
        fc_targ = find_failure_threshold(fr_targ, lc_targ)
        axs[0].plot(fr_rand, lc_rand, label=f"{name} (fc={fc_rand:.2f})")
        axs[1].plot(fr_targ, lc_targ, label=f"{name} (fc={fc_targ:.2f})")

    axs[0].set_title('Random Node Removal')
    axs[0].set_xlabel('Fraction Removed')
    axs[0].set_ylabel('Normalized Largest Component')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Targeted Node Removal')
    axs[1].set_xlabel('Fraction Removed')
    axs[1].set_ylabel('Normalized Largest Component')
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

    # Degree distributions plots
    plot_degree_distribution(
        [(ba_data[0], ba_data[1]), (ba_rand_data[0], ba_rand_data[1]), (ba_targ_data[0], ba_targ_data[1])],
        labels=["BA Original", "BA Random 20%", "BA Targeted 20%"],
        title="BA Degree Distribution"
    )
    plot_degree_distribution(
        [(er_data[0], er_data[1]), (er_rand_data[0], er_rand_data[1]), (er_targ_data[0], er_targ_data[1])],
        labels=["ER Original", "ER Random 20%", "ER Targeted 20%"],
        title="ER Degree Distribution"
    )

    # Network visualizations
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(BA, seed=42)
    plt.subplot(2,2,1)
    nx.draw(BA, pos, node_size=[d*10 for _,d in BA.degree()], node_color='skyblue', edge_color='gray', alpha=0.6)
    plt.title('BA Original')

    plt.subplot(2,2,2)
    pos = nx.spring_layout(BA_targ, seed=42)
    nx.draw(BA_targ, pos, node_size=[d*10 for _,d in BA_targ.degree()], node_color='red', edge_color='gray', alpha=0.6)
    plt.title('BA 20% Targeted Removal')

    plt.subplot(2,2,3)
    pos = nx.spring_layout(ER, seed=42)
    nx.draw(ER, pos, node_size=[d*10 for _,d in ER.degree()], node_color='skyblue', edge_color='gray', alpha=0.6)
    plt.title('ER Original')

    plt.subplot(2,2,4)
    pos = nx.spring_layout(ER_targ, seed=42)
    nx.draw(ER_targ, pos, node_size=[d*10 for _,d in ER_targ.degree()], node_color='red', edge_color='gray', alpha=0.6)
    plt.title('ER 20% Targeted Removal')

    plt.tight_layout()
    plt.show()
    
#DIrectedd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit
import random

def is_directed_graph(G):
    """Check if a graph is directed."""
    return G.is_directed()

def critical_threshold(G):
    """Compute <k^2> / <k>. For directed graphs, use total degree (in+out) per node."""
    degs = []
    if is_directed_graph(G):
        for n in G.nodes():
            d = G.in_degree(n) + G.out_degree(n)
            degs.append(d)
    else:
        for _, d in G.degree():
            degs.append(d)

    if len(degs) > 0:
        k_avg = np.mean(degs)
        k2_avg = np.mean(np.square(degs))
    else:
        k_avg = 0
        k2_avg = 0

    if k_avg > 0:
        thr = k2_avg / k_avg
    else:
        thr = np.inf

    print(f"    -> ⟨k⟩ = {k_avg:.4f}, ⟨k²⟩ = {k2_avg:.4f}, Critical threshold ⟨k²⟩/⟨k⟩ = {thr:.4f}")
    return thr

def compute_metrics_and_distribution(G, label="Network"):
    """Compute network metrics, degree distribution, and power-law fit."""
    print(f"\nComputing metrics for: {label}")
    metrics = {}
    directed = is_directed_graph(G)

    # --- Path Length and Diameter ---
    if directed:
        if nx.is_weakly_connected(G):
            Gc = G
        else:
            # Get largest weakly connected component
            components = list(nx.weakly_connected_components(G))
            if not components:
                metrics['avg_path_length'] = float('inf')
                metrics['diameter'] = float('inf')
            else:
                Gc = G.subgraph(max(components, key=len)).copy()

        if Gc.number_of_nodes() > 1:
            # Use undirected version for path/diameter as per Script 2
            Gc_undir = Gc.to_undirected()
            metrics['avg_path_length'] = nx.average_shortest_path_length(Gc_undir)
            metrics['diameter'] = nx.diameter(Gc_undir)
        else:
            metrics['avg_path_length'] = 0.0
            metrics['diameter'] = 0

    else: # Undirected
        if nx.is_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
        else:
            # Get largest connected component
            components = list(nx.connected_components(G))
            if not components:
                 metrics['avg_path_length'] = float('inf')
                 metrics['diameter'] = float('inf')
            else:
                Gc = G.subgraph(max(components, key=len)).copy()
                if Gc.number_of_nodes() > 1:
                    metrics['avg_path_length'] = nx.average_shortest_path_length(Gc)
                    metrics['diameter'] = nx.diameter(Gc)
                else:
                    metrics['avg_path_length'] = 0.0
                    metrics['diameter'] = 0


    # "Six Degrees" Check (from Script 2)
    if 'avg_path_length' in metrics and metrics['avg_path_length'] != float('inf'):
        if metrics['avg_path_length'] < 6 and metrics['avg_path_length'] > 0:
            print(f"    -> APL ({metrics['avg_path_length']:.2f}) < 6: Consistent with 'Six Degrees of Separation'.")
        else:
            print(f"    -> APL ({metrics['avg_path_length']:.2f}) ≥ 6: Not consistent with 'Six Degrees of Separation'.")

    # --- Clustering ---
    if directed:
        metrics['avg_clustering'] = nx.average_clustering(G.to_undirected())
    else:
        metrics['avg_clustering'] = nx.average_clustering(G)

    # --- Assortativity ---
    try:
        metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
    except nx.NetworkXError:
        metrics['assortativity'] = float('nan') # Handle cases where it's undefined

    # --- Degrees ---
    if directed:
        degrees = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
    else:
        degrees = [d for _, d in G.degree()]

    metrics['min_degree'] = min(degrees) if degrees else 0
    metrics['max_degree'] = max(degrees) if degrees else 0
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0

    # Critical Threshold (from Script 2)
    metrics['critical_threshold'] = critical_threshold(G)

    # --- Degree distribution (for powerlaw fit and plotting) ---
    hist, bin_edges = np.histogram(degrees, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = hist > 0

    # --- Power-law fit ---
    power_law_stats = {}
    fit = None
    if degrees:
        fit = Fit(degrees, verbose=False)
        power_law_stats['alpha'] = fit.power_law.alpha
        power_law_stats['xmin'] = fit.power_law.xmin
        R, p = fit.distribution_compare('power_law', 'exponential')
        power_law_stats['vs_exponential_R'] = R
        power_law_stats['vs_exponential_p'] = p
    else:
        power_law_stats['alpha'] = power_law_stats['xmin'] = 0
        power_law_stats['vs_exponential_R'] = 0
        power_law_stats['vs_exponential_p'] = 1

    return metrics, (hist[valid], bin_centers[valid], fit, power_law_stats)

def remove_nodes(G, fraction, removal_type='random'):
    """Remove fraction of nodes randomly or targeted (by degree). Handles directed graphs."""
    G_copy = G.copy()
    N = len(G_copy)
    num_remove = int(fraction * N)

    if num_remove == 0:
        return G_copy

    if removal_type == 'targeted':
        if is_directed_graph(G_copy):
            # Use total degree (in+out) for directed graphs
            degrees = {n: G_copy.in_degree(n) + G_copy.out_degree(n) for n in G_copy.nodes()}
        else:
            degrees = dict(G_copy.degree)

        nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)
        nodes_to_remove = nodes_to_remove[:num_remove]
    else: # 'random'
        nodes_to_remove = list(G_copy.nodes())
        random.shuffle(nodes_to_remove)
        nodes_to_remove = nodes_to_remove[:num_remove]

    G_copy.remove_nodes_from(nodes_to_remove)
    return G_copy

def percolation_simulation(G, removal_type='random', steps=50):
    """Simulate percolation and return largest component fraction at each step."""
    N = len(G)
    if N == 0:
        return [0], [0]

    fractions = np.linspace(0, 1, steps)
    largest_components = []
    directed = is_directed_graph(G)

    for f in fractions:
        G_copy = remove_nodes(G, f, removal_type)
        if len(G_copy) == 0:
            largest_components.append(0)
        else:
            if directed:
                components = nx.weakly_connected_components(G_copy)
            else:
                components = nx.connected_components(G_copy)

            try:
                largest = max(len(c) for c in components) / N
            except ValueError: # No components found
                largest = 0
            largest_components.append(largest)

    return fractions, largest_components

def find_failure_threshold(fractions, largest_components, threshold=0.01):
    """Estimate percolation threshold where largest component <= threshold."""
    for i, size in enumerate(largest_components):
        if size <= threshold:
            return fractions[i]
    return 1.0

def plot_degree_comparison(hist_bins_list, labels, title="Degree Distribution Comparison"):
    """Plots pre-computed histogram data on a log-log scale."""
    plt.figure(figsize=(10,5))
    for (hist, bins), label in zip(hist_bins_list, labels):
        plt.loglog(bins, hist, marker='o', linestyle='', label=label)
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.show()

def plot_degree_distribution_and_ccdf(G, bins=30, title="Degree Distribution", show_cumulative=False):
    """Plots histogram and optional CCDF for a single graph."""
    plt.figure(figsize=(8,5))

    # Calculate degrees
    degs = []
    if is_directed_graph(G):
        for n in G.nodes():
            degs.append(G.in_degree(n) + G.out_degree(n))
    else:
        for n, d in G.degree():
            degs.append(d)

    if not degs:
        print("Graph has no nodes/degrees to plot.")
        return

    counts, edges = np.histogram(degs, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(bin_centers, counts, 'o', alpha=0.7, label="Empirical P(k)")

    plt.xlabel("Degree (k)")
    plt.ylabel("Probability Density P(k)")
    plt.title(title)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', ls='--')
    plt.show()

    if show_cumulative:
        # Complementary cumulative distribution
        degs_sorted = np.sort(np.array(degs))
        unique, counts = np.unique(degs_sorted, return_counts=True)
        probs = counts / counts.sum()
        # Complementary cumulative distribution P(K >= k)
        ccdf = 1 - np.cumsum(probs) + probs

        plt.figure(figsize=(6,4))
        plt.loglog(unique, ccdf, 'o-', label="Empirical CCDF")
        plt.xlabel("Degree k")
        plt.ylabel("P(K ≥ k)")
        plt.title("Complementary Cumulative Degree Distribution (CCDF)")
        plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.show()

def compare_with_normal_uniform(G, bins=20):
    """Compare empirical degree distribution with fitted normal & uniform distributions."""
    if is_directed_graph(G):
        degs = np.array([G.in_degree(n) + G.out_degree(n) for n in G.nodes()])
    else:
        degs = np.array([d for _, d in G.degree()])

    if len(degs) == 0:
        print("Cannot compare distributions for an empty graph.")
        return

    mu, sigma = degs.mean(), degs.std()
    dmin, dmax = degs.min(), degs.max()
    n = len(degs)

    # Generate comparison samples with same sample size
    normal_sample = np.random.normal(loc=mu, scale=sigma if sigma>0 else 1, size=n)
    normal_sample = np.clip(normal_sample, a_min=0, a_max=None) # Degrees can't be negative
    uniform_sample = np.random.uniform(low=dmin, high=dmax, size=n)

    plt.figure(figsize=(8,6))
    plt.hist(degs, bins=bins, density=True, alpha=0.6, label="Empirical")
    plt.hist(normal_sample, bins=bins, density=True, alpha=0.5, label=f"Normal (μ={mu:.2f},σ={sigma:.2f})")
    plt.hist(uniform_sample, bins=bins, density=True, alpha=0.5, label=f"Uniform [{dmin},{dmax}]")
    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.title("Degree Distribution: Empirical vs Normal & Uniform")
    plt.legend()
    plt.show()

def generate_networks(n=1000, p_er=0.006, m_ba=3, k_ws=6, p_ws=0.1, seed=None, directed=False):
    """Generate ER, BA, and WS networks with a seed."""
    if directed:
        # ER generator has a built-in 'directed' flag
        ER = nx.erdos_renyi_graph(n, p_er, seed=seed, directed=True)

        # BA and WS generators do not. We create the undirected graph
        # and then convert it to a directed graph with symmetric edges.
        BA_undir = nx.barabasi_albert_graph(n, m_ba, seed=seed)
        BA = BA_undir.to_directed()

        WS_undir = nx.watts_strogatz_graph(n, k_ws, p_ws, seed=seed)
        WS = WS_undir.to_directed()

        print("Networks generated (Directed): ER, BA, WS")

    else:
        ER = nx.erdos_renyi_graph(n, p_er, seed=seed, directed=False)
        BA = nx.barabasi_albert_graph(n, m_ba, seed=seed)
        WS = nx.watts_strogatz_graph(n, k_ws, p_ws, seed=seed)
        print("Networks generated (Undirected): ER, BA, WS")

    return ER, BA, WS

# ===================== Main Analysis =====================
if __name__ == "__main__":
    print("=== Complex Network Robustness and Scale-Free Analysis ===")
    n = 1000
    m_ba = 3
    k_ws = 6 # for WS
    p_er = 2*m_ba/(n-1)  # ER probability to match BA average degree ~6
    fraction_remove = 0.2  # 20% nodes removed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # Generate networks
    ER, BA, WS = generate_networks(n=n, p_er=p_er, m_ba=m_ba, k_ws=k_ws, seed=SEED, directed=True)
    # ER, BA, WS = generate_networks(n=n, p_er=p_er, m_ba=m_ba, seed=SEED)
    print("Networks generated: ER, BA, WS")

    # Compute metrics before removal
    ba_metrics, ba_data = compute_metrics_and_distribution(BA, "BA Original")
    er_metrics, er_data = compute_metrics_and_distribution(ER, "ER Original")
    ws_metrics, ws_data = compute_metrics_and_distribution(WS, "WS Original")

    # Node removal: Random and Targeted
    BA_rand = remove_nodes(BA, fraction_remove, 'random')
    BA_targ = remove_nodes(BA, fraction_remove, 'targeted')
    ER_rand = remove_nodes(ER, fraction_remove, 'random')
    ER_targ = remove_nodes(ER, fraction_remove, 'targeted')

    # Compute metrics after removal
    ba_rand_metrics, ba_rand_data = compute_metrics_and_distribution(BA_rand, "BA Random Removal")
    ba_targ_metrics, ba_targ_data = compute_metrics_and_distribution(BA_targ, "BA Targeted Removal")
    er_rand_metrics, er_rand_data = compute_metrics_and_distribution(ER_rand, "ER Random Removal")
    er_targ_metrics, er_targ_data = compute_metrics_and_distribution(ER_targ, "ER Targeted Removal")

    # Print summary metrics
    print("\n--- Summary Metrics ---")
    print("\nBA Original Metrics:", ba_metrics)
    print("BA 20% Random Removal Metrics:", ba_rand_metrics)
    print("BA 20% Targeted Removal Metrics:", ba_targ_metrics)
    print("\nER Original Metrics:", er_metrics)
    print("ER 20% Random Removal Metrics:", er_rand_metrics)
    print("ER 20% Targeted Removal Metrics:", er_targ_metrics)
    print("\n--- End Summary Metrics ---")


    # --- Percolation simulations ---
    print("\nRunning Percolation Simulations...")
    nets = {'ER': ER, 'BA': BA, 'WS': WS}
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    for name, G_net in nets.items():
        fr_rand, lc_rand = percolation_simulation(G_net, 'random')
        fr_targ, lc_targ = percolation_simulation(G_net, 'targeted')
        fc_rand = find_failure_threshold(fr_rand, lc_rand)
        fc_targ = find_failure_threshold(fr_targ, lc_targ)
        axs[0].plot(fr_rand, lc_rand, label=f"{name} (fc={fc_rand:.2f})")
        axs[1].plot(fr_targ, lc_targ, label=f"{name} (fc={fc_targ:.2f})")

    axs[0].set_title('Random Node Removal')
    axs[0].set_xlabel('Fraction Removed')
    axs[0].set_ylabel('Normalized Largest Component')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title('Targeted Node Removal')
    axs[1].set_xlabel('Fraction Removed')
    axs[1].set_ylabel('Normalized Largest Component')
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

    # --- Degree distributions plots (from Script 1) ---
    plot_degree_comparison(
        [(ba_data[0], ba_data[1]), (ba_rand_data[0], ba_rand_data[1]), (ba_targ_data[0], ba_targ_data[1])],
        labels=["BA Original", "BA Random 20%", "BA Targeted 20%"],
        title="BA Degree Distribution (Log-Binned)"
    )
    plot_degree_comparison(
        [(er_data[0], er_data[1]), (er_rand_data[0], er_rand_data[1]), (er_targ_data[0], er_targ_data[1])],
        labels=["ER Original", "ER Random 20%", "ER Targeted 20%"],
        title="ER Degree Distribution (Log-Binned)"
    )

    # --- New Plots (from Script 2) ---
    print("\nShowing CCDF and Comparison Plots...")
    # Plot CCDF for the scale-free (BA) network
    plot_degree_distribution_and_ccdf(BA,
                                      bins=30,
                                      title="BA Original Degree Distribution",
                                      show_cumulative=True)

    # Compare ER network to Normal/Uniform
    compare_with_normal_uniform(ER, bins=20)


    # --- Network visualizations ---
    print("\nGenerating Network Visualizations...")
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(BA, seed=SEED)
    plt.subplot(2,2,1)
    nx.draw(BA, pos, node_size=[d*10 for _,d in BA.degree()], node_color='skyblue', edge_color='gray', alpha=0.6)
    plt.title('BA Original')

    plt.subplot(2,2,2)
    # Recalculate layout for the removed graph for clarity
    pos_targ = nx.spring_layout(BA_targ, seed=SEED)
    nx.draw(BA_targ, pos_targ, node_size=[d*10 for _,d in BA_targ.degree()], node_color='red', edge_color='gray', alpha=0.6)
    plt.title('BA 20% Targeted Removal')

    plt.subplot(2,2,3)
    pos_er = nx.spring_layout(ER, seed=SEED)
    nx.draw(ER, pos_er, node_size=[d*10 for _,d in ER.degree()], node_color='skyblue', edge_color='gray', alpha=0.6)
    plt.title('ER Original')

    plt.subplot(2,2,4)
    pos_er_targ = nx.spring_layout(ER_targ, seed=SEED)
    nx.draw(ER_targ, pos_er_targ, node_size=[d*10 for _,d in ER_targ.degree()], node_color='red', edge_color='gray', alpha=0.6)
    plt.title('ER 20% Targeted Removal')

    plt.tight_layout()
    plt.show()

    print("\n=== Analysis Complete ===")


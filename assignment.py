"""
NETWORK SCIENCE
Notes: change parameters in the bottom block to run different tasks.
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, deque
import random
import os
import warnings
from powerlaw import Fit # Make sure powerlaw is imported

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_graph_auto(path):
    """Load graph from path, auto-detect format by extension.
    Supports: .gml, .csv (edge list), .txt (edgelist), .edgelist, .adjlist
    If unknown, attempt to parse as edge list."""

    ext = os.path.splitext(path)[1].lower()

    if ext == ".gml":
        G = nx.read_gml(path)
        print(f"Loaded GML: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        return G

    elif ext in [".csv"]:
        try:
            df = pd.read_csv(path, index_col=0) # keep index for adjacency matrix check
        except Exception:
             df = pd.read_csv(path) # Fallback if no index col

        #Case 1: adjacency matrix (square & symmetric structure)
        if df.shape[0] == df.shape[1] and np.issubdtype(df.dtypes.values[0], np.number):
            try:
                G = nx.from_pandas_adjacency(df)
                print(f"Loaded CSV adjacency matrix: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
                return G
            except Exception:
                pass # fallback to edge list if it fails

        #Case 2: edge list (at least 2 columns)
        df = pd.read_csv(path) # reload clean without index
        if df.shape[1] >= 2:
            src, tgt = df.columns[0], df.columns[1]
            G = nx.from_pandas_edgelist(
                df, source=src, target=tgt,
                create_using=nx.DiGraph() if 'directed' in df.columns else nx.Graph()
            )
            print(f"Loaded CSV edge list: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            return G
        else:
            raise ValueError("CSV must have at least two columns (source, target).")


    elif ext in [".txt", ".edgelist"]:
        lines = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if line.startswith('#'): # Skip comment lines
                    continue
                if i >= 5: # only check first 5 lines
                    break
                parts = line.strip().split()
                lines.append(parts)

        # If any line has more than 2 entries -> treat as adjlist
        is_adjlist = False
        for parts in lines:
            if len(parts) > 2:
                is_adjlist = True
                break

        if is_adjlist:
            G = nx.read_adjlist(path)
            print(f"Loaded adjlist (detected in .txt): nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        else:
            # Read edgelist, ignoring comments
            G = nx.read_edgelist(path, data=False, comments='#')
            print(f"Loaded edge list: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        return G
    else:
        # Try GML first, then fallback to edgelist
        try:
            G = nx.read_gml(path)
            print("Loaded as GML.")
            return G
        except Exception:
            try:
                G = nx.read_edgelist(path, comments='#')
                print("Loaded as edgelist fallback.")
                return G
            except Exception as e:
                raise ValueError(f"Unsupported or unreadable file format: {e}")

def export_graph(G, name="exported_graph"):
    """Export graph in GML, edgelist, adjlist formats."""
    nx.write_gml(G, f"{name}.gml")
    nx.write_edgelist(G, f"{name}.edgelist")
    nx.write_adjlist(G, f"{name}.adjlist")
    print(f"Exported: {name}.gml, {name}.edgelist, {name}.adjlist")

# -----------------------
# 2) GRAPH GENERATORS
# -----------------------
def generate_er(n, p, seed=SEED):
    return nx.erdos_renyi_graph(n, p, seed=seed)

def generate_ba(n, m, seed=SEED):
    return nx.barabasi_albert_graph(n, m, seed=seed)

def generate_ws(n, k, p, seed=SEED):
    return nx.watts_strogatz_graph(n, k, p, seed=seed)

def generate_powerlaw_configuration(n, gamma, ensure_simple=True):
    # Use powerlaw_sequence to get floats, convert to integers (>=1), then make even sum
    seq = nx.utils.powerlaw_sequence(n, gamma)
    seq = [max(1, int(round(x))) for x in seq]
    if sum(seq) % 2 == 1:
        seq[np.random.randint(0, n)] += 1
    G_multi = nx.configuration_model(seq, create_using=None)
    if ensure_simple:
        G = nx.Graph(G_multi) # removes parallel edges, keeps self-loops as edges (we remove them)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    else:
        return G_multi # Return the graph with self-loops/multi-edges

def generate_stochastic_block_model(nodes_by_color, connection_probs):
    """
    Generic Stochastic Block Model (Color Model) generator for Assignment 1.

    Args:
        nodes_by_color (dict): e.g., {'red': 50, 'blue': 50}
        connection_probs (dict): e.g., {('red', 'red'): 0.1, ('red', 'blue'): 0.01}

    Returns:
        nx.Graph
    """
    G = nx.Graph()
    node_list = [] # Will store tuples of (node_id, color)

    # Create node list and add nodes to graph with color attribute
    current_id = 0
    for color, num_nodes in nodes_by_color.items():
        for _ in range(num_nodes):
            G.add_node(current_id, color=color)
            node_list.append((current_id, color))
            current_id += 1

    # Add edges based on probabilities
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            id_i, color_i = node_list[i]
            id_j, color_j = node_list[j]

            # Find connection probability
            # Check for (color_i, color_j)
            prob = connection_probs.get((color_i, color_j), 0)
            if (color_i, color_j) not in connection_probs:
                # If not found, check for (color_j, color_i)
                prob = connection_probs.get((color_j, color_i), 0)

            if random.random() < prob:
                G.add_edge(id_i, id_j)

    return G

# -----------------------
# 3) METRICS & HELPERS
# -----------------------
def is_directed_graph(G):
    return G.is_directed()

def degree_stats(G):
    """Return degree list and summary stats depending on directedness."""
    if is_directed_graph(G):
        in_deg = []
        for _, d in G.in_degree():
            in_deg.append(d)
        out_deg = []
        for _, d in G.out_degree():
            out_deg.append(d)
        return {"in": in_deg, "out": out_deg}
    else:
        deg = []
        for _, d in G.degree():
            deg.append(d)
        return {"deg": deg}

def average_degree(G):
    if is_directed_graph(G):
        # For directed graphs, average in-degree == average out-degree == m/n
        return G.number_of_edges() / G.number_of_nodes()
    else:
        deg = []
        for _, d in G.degree():
            deg.append(d)
        return np.mean(deg)

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
    print(f"⟨k⟩ = {k_avg:.4f}, ⟨k²⟩ = {k2_avg:.4f}, Critical threshold ⟨k²⟩/⟨k⟩ = {thr:.4f}")
    return thr

def avg_clustering(G):
    """Compute average clustering coefficient (directed → treat as undirected)."""
    if is_directed_graph(G):
        avg = nx.average_clustering(G.to_undirected())
    else:
        avg = nx.average_clustering(G)

    print(f"Average clustering coefficient: {avg:.4f}")
    return avg

def avg_path_and_diameter(G):
    """Compute average shortest path length and diameter on largest connected/weakly connected component."""
    if is_directed_graph(G):
        # use largest weakly connected component
        components = list(nx.weakly_connected_components(G))
        if not components:
            print("Graph is empty.")
            return np.nan, np.nan
        comp = max(components, key=len)
        Gc = G.subgraph(comp).copy()
    else:
        components = list(nx.connected_components(G))
        if not components:
            print("Graph is empty.")
            return np.nan, np.nan
        comp = max(components, key=len)
        Gc = G.subgraph(comp).copy()

    if Gc.number_of_nodes() <= 1:
        print("Component too small to compute path stats.")
        return np.nan, np.nan

    try:
        if is_directed_graph(G):
            # shortest path length treating as undirected for distance measure
            Gc_undir = Gc.to_undirected()
            apl = nx.average_shortest_path_length(Gc_undir)
            diam = nx.diameter(Gc_undir)
        else:
            apl = nx.average_shortest_path_length(Gc)
            diam = nx.diameter(Gc)

        # Add six degrees of separation check here
        print(f"Average shortest path length (LCC): {apl:.4f}, Diameter (LCC): {diam}")
        if apl < 6 and apl > 0:
            print("→ APL < 6: Consistent with 'Six Degrees of Separation'.")
        else:
            print("→ APL ≥ 6: Not consistent with 'Six Degrees of Separation'.")

        return apl, diam
    except Exception as e:
        print("Error computing path/diameter (graph may be disconnected?):", e)
        return np.nan, np.nan

# -----------------------
# 4) DEGREE DISTRIBUTION & COMPARISONS
# -----------------------
def plot_degree_distribution(G, bins=30, title="Degree Distribution", show_cumulative=False):
    plt.figure(figsize=(8,5))
    # Calculate degrees without list comprehension
    degs = []
    if is_directed_graph(G):
        for n in G.nodes():
            degs.append(G.in_degree(n) + G.out_degree(n))
    else:
        for n, d in G.degree():
            degs.append(d)

    if not degs:
        print("Cannot plot degree distribution for empty graph.")
        plt.close()
        return

    counts, edges, patches = plt.hist(degs, bins=bins, density=True, alpha=0.7, label="Empirical deg")
    plt.xlabel("Degree")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.show()

    if show_cumulative:
        # Complementary cumulative distribution
        degs_sorted = np.sort(degs)
        unique, counts = np.unique(degs_sorted, return_counts=True)
        probs = counts / counts.sum()
        ccdf = []
        # Complementary cumulative distribution P(K >= k)
        ccdf = 1 - np.cumsum(probs) + probs
        plt.figure(figsize=(6,4))
        plt.loglog(unique, ccdf, 'o-')
        plt.xlabel("Degree k")
        plt.ylabel("P(K ≥ k)")
        plt.title("Complementary Cumulative Degree Distribution (CCDF)")
        plt.show()

def compare_with_normal_uniform(G, bins=20):
    """Compare empirical degree distribution with fitted normal & uniform distributions (matched to empirical mean/std/range)."""
    if is_directed_graph(G):
        degs = np.array([G.in_degree(n) + G.out_degree(n) for n in G.nodes()])
    else:
        degs = np.array([d for _, d in G.degree()])

    if len(degs) == 0:
        print("Cannot compare distributions for empty graph.")
        return

    mu, sigma = degs.mean(), degs.std()
    dmin, dmax = degs.min(), degs.max()
    n = len(degs)

    # Generate comparison samples with same sample size
    normal_sample = np.random.normal(loc=mu, scale=sigma if sigma>0 else 1, size=n)
    normal_sample = np.clip(normal_sample, a_min=0, a_max=None)
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

# -----------------------
# 5) ASSIGNMENT-SPECIFIC SIMULATIONS
# -----------------------
# 5.1 Attack simulations (Programming Assignment 3)
def simulate_attack(G, criterion='degree', fractions=np.linspace(0,0.5,11), directed=False):
    """Simulate removal of fraction f of nodes ranked by criterion (degree or clustering).
       Returns list of normalized giant component sizes for each fraction f.
    """
    G0 = G.copy()
    N = G0.number_of_nodes()
    sizes = []
    if criterion == 'degree':
        if directed:
            # use total degree for ranking
            nodes_sorted = sorted(G0.nodes(), key=lambda n: (G0.in_degree(n) + G0.out_degree(n)), reverse=True)
        else:
            nodes_sorted = sorted(G0.nodes(), key=lambda n: G0.degree(n), reverse=True)
    elif criterion == 'clustering':
        cl = nx.clustering(G0.to_undirected()) if directed else nx.clustering(G0)
        nodes_sorted = sorted(G0.nodes(), key=lambda n: cl.get(n, 0), reverse=True)
    else:
        raise ValueError("criterion must be 'degree' or 'clustering'")

    for f in fractions:
        k = int(np.round(f * N))
        Gtemp = G0.copy()
        remove_nodes = nodes_sorted[:k]
        Gtemp.remove_nodes_from(remove_nodes)
        if Gtemp.number_of_nodes() == 0:
            sizes.append(0.0)
        else:
            if is_directed_graph(Gtemp):
                comp = max(nx.weakly_connected_components(Gtemp), key=len) if nx.number_weakly_connected_components(Gtemp) > 0 else set()
            else:
                comp = max(nx.connected_components(Gtemp), key=len) if nx.number_connected_components(Gtemp) > 0 else set()
            sizes.append(len(comp) / N)
    return sizes

# 5.2 Sandpile simulation (Programming Assignment 4)
def simulate_sandpile(G, steps=2000):
    """Sandpile avalanche model: bucket capacity = node degree.
       Returns list of avalanche sizes (number of toppled nodes) per perturbation.
    """
    # Work with a simple graph structure for neighbors
    G_use = G.to_undirected() if is_directed_graph(G) else G

    if G_use.number_of_nodes() == 0:
        print("Cannot run sandpile on empty graph.")
        return []

    buckets = {n: max(1, G_use.degree(n)) for n in G_use.nodes()} # bucket size at least 1
    grains = {n: 0 for n in G_use.nodes()}
    avalanches = []

    nodes_list = list(G_use.nodes())
    for t in range(steps):
        i = random.choice(nodes_list)
        grains[i] += 1
        unstable = deque([i]) if grains[i] >= buckets[i] else deque()
        toppled = set()
        while unstable:
            u = unstable.popleft()
            if u not in grains: continue # Node might have been removed

            if grains[u] >= buckets[u]:
                toppled.add(u)
                num = grains[u]
                grains[u] = 0
                neighbors = list(G_use.neighbors(u))
                if len(neighbors) == 0:
                    continue
                # distribute equally (integer division) and random remainder distribution
                per = num // len(neighbors)
                rem = num % len(neighbors)
                for nb in neighbors:
                    grains[nb] += per
                # distribute remainder to random neighbors
                for r in range(rem):
                    nb = random.choice(neighbors)
                    grains[nb] += 1
                for nb in neighbors:
                    if grains[nb] >= buckets[nb] and nb not in toppled:
                        unstable.append(nb)
        if len(toppled) > 0:
            avalanches.append(len(toppled))
    return avalanches

# -----------------------
# 6) VISUALIZATION HELPERS
# -----------------------
def visualize_graph(G, title="Graph", node_size=50, with_labels=False, layout_seed=SEED):
    plt.figure(figsize=(6,6))

    if G.number_of_nodes() > 2000:
       pos = nx.random_layout(G, seed=layout_seed)
    else:
       pos = nx.spring_layout(G, seed=layout_seed)

    nx.draw(G, pos, node_size=node_size, with_labels=with_labels, alpha=0.7, edge_color='gray')
    plt.title(title)
    plt.show()

def plot_avalanche_sizes(avalanche_list, title="Avalanche size distribution"):
    if len(avalanche_list) == 0:
        print("No avalanches recorded.")
        return
    plt.figure(figsize=(7,5))

    # Use numpy histogram to get bins and counts for log plotting
    counts, bin_edges = np.histogram(avalanche_list, bins=np.logspace(np.log10(min(avalanche_list)), np.log10(max(avalanche_list)), 30), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.loglog(bin_centers[counts>0], counts[counts>0], 'o', label='Simulated data')

    # Optional: Fit a power law to the tail
    try:
        fit = Fit(avalanche_list, verbose=False)
        fit.power_law.plot_pdf(color='r', linestyle='--', label=f'Power law (α={fit.power_law.alpha:.2f})')
        plt.legend()
    except Exception as e:
        print(f"Could not fit powerlaw: {e}")

    plt.xlabel("Avalanche size")
    plt.ylabel("Probability P(s)")
    plt.title(title)
    plt.grid(True, which='both', ls='--')
    plt.show()

def plot_attack_results(fractions, sizes_degree, sizes_clustering, title="Attack simulation"):
    plt.figure(figsize=(7,5))
    plt.plot(fractions, sizes_degree, 'o-', label='Remove by degree')
    plt.plot(fractions, sizes_clustering, 's-', label='Remove by clustering')
    plt.xlabel("Fraction nodes removed")
    plt.ylabel("Normalized giant component size")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

# -----------------------
# 7) UTILITY: summary printed report
# -----------------------
def full_report(G, name="Graph"):
    print("\n========== SUMMARY REPORT ==========")
    print(f"Name: {name}")
    print("Directed:", is_directed_graph(G))
    N = G.number_of_nodes()
    M = G.number_of_edges()
    print("Nodes:", N, "Edges:", M)
    if N == 0:
        print("====================================\n")
        return

    # Degrees without list comprehensions
    if is_directed_graph(G):
        indeg = []
        outdeg = []
        for _, d in G.in_degree():
            indeg.append(d)
        for _, d in G.out_degree():
            outdeg.append(d)
        print(f"Average in-degree: {np.mean(indeg):.4f}, Average out-degree: {np.mean(outdeg):.4f}")
    else:
        degs = []
        for _, d in G.degree():
            degs.append(d)
        print(f"Average degree: {np.mean(degs):.4f}")

    # Clustering
    avg_clustering(G)

    # Threshold
    critical_threshold(G)

    # Path & diameter
    avg_path_and_diameter(G)

    # Visualize small graphs
    if G.number_of_nodes() <= 1000:
        visualize_graph(G, title=f"{name} visualization")

    # Plot degree distribution and comparisons
    plot_degree_distribution(G, bins=30, title=f"{name} Degree Distribution", show_cumulative=True)
    compare_with_normal_uniform(G, bins=30)

    print("====================================\n")
# -----------------------
# 8) MAIN: examples & quick experiments
# -----------------------

if __name__ == "__main__":
    # ---------- SETTINGS ----------
    # Choose one: "er", "ws", "ba", "scalefree", "dataset", "assignment1", "assignment2", "assignment3", "assignment4"

    mode = "assignment3" # <--- CHANGE THIS TO SWITCH MODE

    # Common parameters
    N = 500 # number of nodes
    N_small = 500
    N_large = 10000
    avg_k = 4 # target average degree
    p = avg_k / (N - 1) # for ER
    m = 3 # for BA (number of edges to attach per new node)
    k = 4 # each node connected to k neighbors in WS
    beta = 0.1 # rewiring prob for WS
    gamma = 2.5 # exponent for power-law configuration model

    #when datset is given
    if mode == "dataset":
         # Simple demo: load sample GML if present, otherwise generate ER
        sample_path = "facebook_combined.txt" # e.g., /content/facebook_combined.txt
        if os.path.exists(sample_path):
            G = load_graph_auto(sample_path)
        else:
            print(f"Warning: Dataset '{sample_path}' not found. Generating sample ER graph.")
            G = generate_er(N, p)
        full_report(G, name="Dataset Graph")
        if os.path.exists(sample_path):
            export_graph(G, name=f"exported_{os.path.splitext(os.path.basename(sample_path))[0]}")
        else:
            export_graph(G, name="exported_ER")

        # Same size & avg degree as dataset
        n = G.number_of_nodes()
        if is_directed_graph(G):
             avg_k = G.number_of_edges() / n
        else:
             avg_k = np.mean([deg for _, deg in G.degree()])

        p = avg_k / (n-1)
        k = int(round(avg_k))

        if k % 2 == 1:
            k += 1 # WS requires even k

        p_ws = 0.1 # rewiring probability
        er_graph = generate_er(n, p)
        sf_graph = generate_powerlaw_configuration(n, gamma=2.5)
        ws_graph = generate_ws(n, k, p_ws)
        full_report(er_graph, name="ER reference")
        full_report(sf_graph, name="Scale-Free reference")
        full_report(ws_graph, name="WS reference")

    elif mode == "er":
        print("\n=== ER RANDOM GRAPH ===")
        G = generate_er(N, p)
        full_report(G, name=f"ER (N={N}, p={p:.3f})")

    elif mode == "ws":
        print("\n=== WATTS–STROGATZ SMALL WORLD ===")
        G = generate_ws(N, k, beta)
        full_report(G, name=f"WS (N={N}, k={k}, beta={beta})")

    elif mode == "ba":
        print("\n=== BARABÁSI–ALBERT SCALE-FREE ===")
        G = generate_ba(N, m)
        full_report(G, name=f"BA (N={N}, m={m})")

    elif mode == "scalefree":
        print("\n=== SCALE-FREE NETOWRK ===")
        G = generate_powerlaw_configuration(N, gamma)
        full_report(G, name=f"ScaleFree Network (N={N}, γ={gamma})")

    elif mode == "assignment1":
        # --- Q1 ---
        print("\n--- Assignment 1, Q1: ER Graph Visualization ---")
        N_q1 = 500
        avg_degrees = [0.8, 1, 8]
        for k_q1 in avg_degrees:
            p_q1 = k_q1 / (N_q1 - 1)
            G_q1 = generate_er(N_q1, p_q1)
            print(f"\nER N={N_q1}, <k>={k_q1} (p={p_q1:.6f}): nodes={G_q1.number_of_nodes()}, edges={G_q1.number_of_edges()}")
            # Use full_report to visualize and analyze
            full_report(G_q1, name=f"ER_k_{k_q1}")

        # --- Q2 & Q3 ---
        print("\n--- Assignment 1, Q2 & Q3: Stochastic Block Model (Simulation) ---")
        print("NOTE: This code *generates* the networks. It does not solve the theoretical parts (2a, 2b, 2c, 3a, 3b).")

        # --- Q2 Example (snobbish) ---
        N_q2 = 100 # Using smaller N for visualization
        p_q2 = 0.1  # High same-color prob
        q_q2 = 0.01 # Low different-color prob
        G_q2 = generate_stochastic_block_model(
            nodes_by_color={'red': N_q2, 'blue': N_q2},
            connection_probs={
                ('red', 'red'): p_q2,
                ('blue', 'blue'): p_q2,
                ('red', 'blue'): q_q2
            }
        )
        full_report(G_q2, name=f"Q2 Snobbish Model (p={p_q2}, q={q_q2})")

        # --- Q3 Example ---
        N_red = 50
        N_blue = 50
        N_purple = 20 # 20 purple nodes
        p_q3 = 0.1 # This is 'p' in the prompt

        G_q3 = generate_stochastic_block_model(
            nodes_by_color={'red': N_red, 'blue': N_blue, 'purple': N_purple},
            connection_probs={
                ('red', 'red'): p_q3,      # Red connect to Red
                ('blue', 'blue'): p_q3,     # Blue connect to Blue
                ('red', 'blue'): 0.0,       # Red/Blue do not connect (q=0)
                ('purple', 'red'): p_q3,    # Purple connect to Red
                ('purple', 'blue'): p_q3    # Purple connect to Blue
                # ('purple', 'purple') prob is 0 by default, as it's not specified
            }
        )
        full_report(G_q3, name=f"Q3 Purple Model (p={p_q3}, q=0)")

    elif mode == "assignment2":
        # Programming Assignment 2: power-law seq and BA snapshots

        # --- Q1 ---
        print("\n--- Assignment 2, Q1: Configuration Model Analysis ---")
        for gamma_q2 in [2.2, 3.0]:
            # Use a smaller N for a quick demo, 10^5 is very slow
            for N_q2 in [1000, 10000]:
                print(f"Analyzing N={N_q2}, gamma={gamma_q2}")

                # Step 1: Generate the sequence
                seq = nx.utils.powerlaw_sequence(N_q2, gamma_q2)
                seq = [max(1, int(round(x))) for x in seq]
                if sum(seq) % 2 == 1:
                    seq[np.random.randint(0, N_q2)] += 1

                # Step 2: Create the multi-graph (as requested)
                G_multi = nx.configuration_model(seq, create_using=None)

                # Step 3: Count self-loops and multi-links
                total_edges = G_multi.number_of_edges()
                if total_edges == 0:
                    print("  Graph has 0 edges.")
                    continue

                num_selfloops = nx.number_of_selfloops(G_multi)

                # Find "extra" edges from multi-links
                multi_edge_count = 0
                nodes_list = list(G_multi.nodes())
                edges_seen = set()
                for u, v in G_multi.edges():
                    if u > v: u, v = v, u # Standardize edge tuple
                    if (u, v) in edges_seen and u != v:
                        multi_edge_count += 1
                    elif u != v:
                        edges_seen.add((u,v))

                perc_selfloop = (num_selfloops / total_edges) * 100
                perc_multilink = (multi_edge_count / total_edges) * 100

                print(f"  Total Edges (in G_multi): {total_edges}")
                print(f"  Self-loops: {num_selfloops} ({perc_selfloop:.2f}%)")
                print(f"  Multi-links (extra edges): {multi_edge_count} ({perc_multilink:.2f}%)")

                # Step 4: Create the simple graph for plotting
                G = nx.Graph(G_multi)
                G.remove_edges_from(nx.selfloop_edges(G))
                plot_degree_distribution(G, title=f"Config model (Simple): N={N_q2}, γ={gamma_q2}", bins=50, show_cumulative=True)

        # --- Q2 ---
        print("\n--- Assignment 2, Q2: BA Snapshots & Clustering ---")
        N_full = 10000
        m = 4

        # Q2a,b,c: Plot snapshots
        for ns in [100, 1000, 10000]:
            Gb = generate_ba(ns, m)
            plot_degree_distribution(Gb, title=f"BA snapshot N={ns}", show_cumulative=True)

        # Q2d: Plot clustering vs N
        Ns = [100, 200, 500, 1000, 2000, 5000, 10000]
        clust_values = []
        for n_snap in Ns:
            Gb = generate_ba(n_snap, m)
            clust_values.append(nx.average_clustering(Gb))

        plt.figure();
        plt.plot(Ns, clust_values, 'o-');
        plt.xscale('log');
        plt.xlabel('N (Number of Nodes)');
        plt.ylabel('Average Clustering Coefficient');
        plt.title('BA Clustering vs. N (m=4)');
        plt.grid(True)
        plt.show()

    elif mode == "assignment3":
        # Programming Assignment 3: attack simulations
        N_q3 = 10000
        m_q3 = 4
        print(f"Generating config model (N={N_q3}, γ=2.5) ...")
        G_config = generate_powerlaw_configuration(N_q3, gamma=2.5, ensure_simple=True)

        fractions = np.linspace(0, 0.5, 11)
        sizes_deg = simulate_attack(G_config, 'degree', fractions)
        sizes_clu = simulate_attack(G_config, 'clustering', fractions)
        print("Config model attack results (degree vs clustering):", sizes_deg, sizes_clu)
        plot_attack_results(fractions, sizes_deg, sizes_clu, title="Config model attack")

        # Hierarchical (powerlaw_cluster_graph)
        print(f"Generating hierarchical model (N={N_q3}, m={m_q3}) ...")
        G_hier = nx.powerlaw_cluster_graph(N_q3, m=m_q3, p=0.1)
        sizes_deg_h = simulate_attack(G_hier, 'degree', fractions)
        sizes_clu_h = simulate_attack(G_hier, 'clustering', fractions)
        plot_attack_results(fractions, sizes_deg_h, sizes_clu_h, title="Hierarchical model attack")

    elif mode == "assignment4":
        # Programming Assignment 4: Sandpile on ER vs Scale-free
        N_q4 = 1000 # smaller for speed
        avg_k_q4 = 2
        p_q4 = avg_k_q4 / (N_q4 - 1)

        print(f"Running Sandpile on ER (N={N_q4}, <k>={avg_k_q4})...")
        G_er = generate_er(N_q4, p_q4)
        aval_er = simulate_sandpile(G_er, steps=2000)
        print(f"ER mean avalanche size: {np.mean(aval_er) if len(aval_er)>0 else 0:.3f}")
        plot_avalanche_sizes(aval_er, title=f"ER avalanches N={N_q4}, <k>={avg_k_q4}")

        # Scale-free via configuration
        print(f"Running Sandpile on Scale-Free (N={N_q4}, γ=2.5)...")
        G_sf = generate_powerlaw_configuration(N_q4, gamma=2.5, ensure_simple=True)
        aval_sf = simulate_sandpile(G_sf, steps=2000)
        print(f"SF mean avalanche size: {np.mean(aval_sf) if len(aval_sf)>0 else 0:.3f}")
        plot_avalanche_sizes(aval_sf, title=f"SF avalanches N={N_q4}, γ=2.5")

    else:
        print("Unknown mode. Set mode to one of: er, ws, ba, scalefree, dataset, assignment1, assignment2, assignment3, assignment4")
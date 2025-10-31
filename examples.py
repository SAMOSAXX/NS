
# 8) MAIN: examples & quick experiments
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, deque
import random
import os
import warnings



if __name__ == "__main__":
    # ---------- SETTINGS ----------
    # Choose one: "er", "ws", "ba", "config", "example", "from_file", "assignment1", "assignment2", "assignment3", "assignment4"
    mode = "er"  # <--- CHANGE THIS TO SWITCH MODE

    # Common parameters
    N = 500   # number of nodes
    N_small = 500
    N_large = 10000
    avg_k = 4 # target average degree
    p = avg_k / (N - 1)  # for ER
    m = 3     # for BA (number of edges to attach per new node)
    k = 4     # each node connected to k neighbors in WS
    beta = 0.1 # rewiring prob for WS
    gamma = 2.5 # exponent for power-law configuration model

    if mode == "er":
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

    elif mode == "config":
        print("\n=== CONFIGURATION MODEL SCALE-FREE ===")
        G = generate_powerlaw_configuration(N, gamma)
        full_report(G, name=f"Config Model (N={N}, γ={gamma})")

    elif mode == "example":
         # Simple demo: load sample GML if present, otherwise generate ER
        sample_path = "/content/network.gml"
        if os.path.exists(sample_path):
            G = load_graph_auto(sample_path)
        else:
            G = generate_er(200, 0.03)
        full_report(G, name="Example Graph")
        export_graph(G, name=f"exported_{os.path.splitext(os.path.basename(path))[0]}")

    elif mode == "assignment1":
        # Programming Assignment 1: ER graphs with different avg degrees
        N = 500
        avg_degrees = [0.8, 1, 8]
        for k in avg_degrees:
            p = k / (N - 1)
            G = generate_er(N, p)
            print(f"\nER N={N}, <k>={k} (p={p:.6f}): nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            full_report(G, name=f"ER_k_{k}")
        # You can add red/blue/purple community code from your assignment as needed.

    elif mode == "assignment2":
        # Programming Assignment 2: power-law seq and BA snapshots
        # 1) Percent multi-links & self-loops as function of N for gamma values
        for gamma in [2.2, 3.0]:
            for N in [1000, 10000, 100000]:
                G = generate_powerlaw_configuration(N, gamma)
                # Rough estimate: self-loops were removed in generation; but if using pure config model,
                # create multi-graph and count self-loops/multi-edges if needed.
                print(f"Generated configuration-based graph N={N}, gamma={gamma}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
                # Show degree distribution
                plot_degree_distribution(G, title=f"Config model: N={N}, γ={gamma}", bins=50, show_cumulative=True)
        # 2) BA network snapshots and clustering vs N
        N_full = 10000
        m = 4
        G_ba = generate_ba(N_full, m)
        for ns in [100, 1000, 10000]:
            # approximate snapshot by building a BA of that size
            Gb = generate_ba(ns, m)
            plot_degree_distribution(Gb, title=f"BA snapshot N={ns}", show_cumulative=True)
        Ns = [100, 200, 500, 1000, 2000, 5000, 10000]
        clust_values = []
        for n in Ns:
            Gb = generate_ba(n, m)
            clust_values.append(nx.average_clustering(Gb))
        plt.figure(); plt.plot(Ns, clust_values, 'o-'); plt.xscale('log'); plt.xlabel('N'); plt.ylabel('Avg clustering'); plt.title('BA clustering vs N'); plt.show()

    elif mode == "assignment3":
        # Programming Assignment 3: attack simulations
        N = 10000
        print("Generating config model (power-law, γ=2.5) ...")
        degrees = nx.utils.powerlaw_sequence(N, 2.5)
        seq = [max(1, int(round(x))) for x in degrees]
        if sum(seq) % 2 == 1: seq[np.random.randint(0, N)] += 1
        G_config = nx.configuration_model(seq)
        G_config = nx.Graph(G_config)
        G_config.remove_edges_from(nx.selfloop_edges(G_config))
        fractions = np.linspace(0, 0.5, 11)
        sizes_deg = simulate_attack(G_config, 'degree', fractions)
        sizes_clu = simulate_attack(G_config, 'clustering', fractions)
        print("Config model attack results (degree vs clustering):", sizes_deg, sizes_clu)
        plot_attack_results(fractions, sizes_deg, sizes_clu, title="Config model attack")

        # Hierarchical (powerlaw_cluster_graph)
        print("Generating hierarchical model ...")
        G_hier = nx.powerlaw_cluster_graph(N, m=4, p=0.1)
        sizes_deg_h = simulate_attack(G_hier, 'degree', fractions)
        sizes_clu_h = simulate_attack(G_hier, 'clustering', fractions)
        plot_attack_results(fractions, sizes_deg_h, sizes_clu_h, title="Hierarchical model attack")

    elif mode == "assignment4":
        # Programming Assignment 4: Sandpile on ER vs Scale-free
        N = 1000  # smaller for speed in exam environment
        avg_k = 2
        p = avg_k / (N - 1)
        G_er = generate_er(N, p)
        aval_er = simulate_sandpile(G_er, steps=2000)
        print(f"ER mean avalanche size: {np.mean(aval_er) if len(aval_er)>0 else 0:.3f}")
        plot_avalanche_sizes(aval_er, title=f"ER avalanches N={N}, <k>={avg_k}")

        # Scale-free via configuration
        seq = generate_powerlaw_configuration(N, gamma=2.5, ensure_simple=True)
        G_sf = seq  # function returned a graph
        aval_sf = simulate_sandpile(G_sf, steps=2000)
        print(f"SF mean avalanche size: {np.mean(aval_sf) if len(aval_sf)>0 else 0:.3f}")
        plot_avalanche_sizes(aval_sf, title=f"SF avalanches N={N}, <k>={avg_k}")

    else:
        print("Unknown mode. Set mode to one of: example, from_file, assignment1, assignment2, assignment3, assignment4")

# 7) UTILITY: summary printed report
#Use functions with other files functions


import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, deque
import random
import os
import warnings

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def full_report(G, name="Graph"):
    print("\n========== SUMMARY REPORT ==========")
    print(f"Name: {name}")
    print("Directed:", nx.is_directed_graph(G))
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
    # Degrees
    if nx.is_directed_graph(G):
        indeg = [d for _, d in G.in_degree()]
        outdeg = [d for _, d in G.out_degree()]
        print(f"Average in-degree: {np.mean(indeg):.4f}, Average out-degree: {np.mean(outdeg):.4f}")
    else:
        degs = [d for _, d in G.degree()]
        print(f"Average degree: {np.mean(degs):.4f}")
    # Clustering
    avg_clustering(G)
    # Threshold
    critical_threshold(G)
    # path & diameter
    avg_path_and_diameter(G)
    # Visualize small
    if G.number_of_nodes() <= 1000:
        visualize_graph(G, title=f"{name} visualization")
    # Plot degree distribution and comparisons
    plot_degree_distribution(G, bins=30, title=f"{name} Degree Distribution", show_cumulative=True)
    compare_with_normal_uniform(G, bins=30)
    print("====================================\n")

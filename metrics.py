# 3) METRICS & HELPERS
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


def is_directed_graph(G):
    return G.is_directed()

def degree_stats(G):
    """Return degree list and summary stats depending on directedness."""
    if is_directed_graph(G):
        in_deg = [d for _, d in G.in_degree()]
        out_deg = [d for _, d in G.out_degree()]
        return {"in": in_deg, "out": out_deg}
    else:
        deg = [d for _, d in G.degree()]
        return {"deg": deg}

def average_degree(G):
    if is_directed_graph(G):
        # For directed graphs, average in-degree == average out-degree == m/n
        in_deg = [d for _, d in G.in_degree()]
        return np.mean(in_deg)
    else:
        deg = [d for _, d in G.degree()]
        return np.mean(deg)

def critical_threshold(G):
    """Compute <k^2> / <k>. For directed graphs, use total degree (in+out) per node."""
    if is_directed_graph(G):
        degs = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
    else:
        degs = [d for _, d in G.degree()]
    k_avg = np.mean(degs) if len(degs) > 0 else 0
    k2_avg = np.mean(np.square(degs)) if len(degs) > 0 else 0
    thr = k2_avg / k_avg if k_avg > 0 else np.inf
    print(f"⟨k⟩ = {k_avg:.4f}, ⟨k²⟩ = {k2_avg:.4f}, Critical threshold ⟨k²⟩/⟨k⟩ = {thr:.4f}")
    return thr

def avg_clustering(G):
    # For directed graphs, networkx.clustering treats them as undirected by default
    avg = nx.average_clustering(G.to_undirected()) if is_directed_graph(G) else nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg:.4f}")
    return avg

def avg_path_and_diameter(G):
    """Compute average shortest path length and diameter on largest connected/weakly connected component."""
    if is_directed_graph(G):
        # use largest weakly connected component
        if nx.is_weakly_connected(G):
            Gc = G
        else:
            comp = max(nx.weakly_connected_components(G), key=len)
            Gc = G.subgraph(comp).copy()
    else:
        if nx.is_connected(G):
            Gc = G
        else:
            comp = max(nx.connected_components(G), key=len)
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
        if apl < 6:
            print("→ APL < 6: Consistent with 'Six Degrees of Separation'.")
        else:
            print("→ APL ≥ 6: Not consistent with 'Six Degrees of Separation'.")

        return apl, diam
    except Exception as e:
        print("Error computing path/diameter:", e)
        return np.nan, np.nan

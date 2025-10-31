

# 4) DEGREE DISTRIBUTION & COMPARISONS
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


def plot_degree_distribution(G, bins=30, title="Degree Distribution", show_cumulative=False):
    plt.figure(figsize=(8,5))
    if nx.is_directed_graph(G):
        degs = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
    else:
        degs = [d for _, d in G.degree()]

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
        ccdf = 1 - np.cumsum(probs) + probs  # P(K >= k)
        plt.figure(figsize=(6,4))
        plt.loglog(unique, ccdf, 'o-')
        plt.xlabel("Degree k")
        plt.ylabel("P(K ≥ k)")
        plt.title("Complementary Cumulative Degree Distribution (CCDF)")
        plt.show()

def compare_with_normal_uniform(G, bins=20):
    """Compare empirical degree distribution with fitted normal & uniform distributions (matched to empirical mean/std/range)."""
    if nx.is_directed_graph(G):
        degs = np.array([G.in_degree(n) + G.out_degree(n) for n in G.nodes()])
    else:
        degs = np.array([d for _, d in G.degree()])

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
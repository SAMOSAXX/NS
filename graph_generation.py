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
        G = nx.Graph(G_multi)  # removes parallel edges, keeps self-loops as edges (we remove them)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    else:
        return G_multi

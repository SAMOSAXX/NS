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


# 1) FILE I/O & LOADING

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
        df = pd.read_csv(path)
        # Try to guess the source and target columns
        if df.shape[1] >= 2:
            src, tgt = df.columns[0], df.columns[1]
            G = nx.from_pandas_edgelist(df, source=src, target=tgt, create_using=nx.DiGraph() if 'directed' in df.columns else nx.Graph())
            print(f"Loaded CSV edge list: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            return G
        else:
            raise ValueError("CSV must have at least two columns (source, target).")

    '''
        elif ext in [".csv"]:
        df = pd.read_csv(path, index_col=0)  # keep index for adjacency matrix check

        #Case 1: adjacency matrix (square & symmetric structure)
        if df.shape[0] == df.shape[1] and np.issubdtype(df.dtypes.values[0], np.number):
            try:
                G = nx.from_pandas_adjacency(df)
                print(f"Loaded CSV adjacency matrix: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
                return G
            except Exception:
                pass  # fallback to edge list if it fails

        #Case 2: edge list (at least 2 columns)
        df = pd.read_csv(path)  # reload clean without index
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
    '''
    elif ext in [".txt", ".edgelist"]:
        G = nx.read_edgelist(path, data=False)
        print(f"Loaded edge list: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        return G
    elif ext == ".adjlist":
        G = nx.read_adjlist(path)
        print(f"Loaded adjlist: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        return G
    else:
        # Try GML first, then fallback to edgelist
        try:
            G = nx.read_gml(path)
            print("Loaded as GML.")
            return G
        except Exception:
            try:
                G = nx.read_edgelist(path)
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

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_config = 10000  # Number of nodes for configuration model
gamma = 2.5  # Power-law exponent
k_min = 3  # Minimum degree
steps = 20  # Number of fractions for removal simulation

def generate_hierarchical(level):
    # Base case: level 1
    G = nx.complete_graph(5)
    central = 0
    external = [1, 2, 3, 4]

    # Iteratively build hierarchy for levels > 1
    for _ in range(2, level + 1):
        G_prev = G.copy()
        central_prev = central
        external_prev = external

        node_offset = len(G)
        new_external = []
        # Create 4 replicas of previous graph
        for _ in range(4):
            H = G_prev.copy()
            mapping = {u: u + node_offset for u in H.nodes()}
            H = nx.relabel_nodes(H, mapping)
            G = nx.union(G, H)

            rep_external = [e + node_offset for e in external_prev]
            new_external += rep_external
            # Connect new externals to old central
            for ex in rep_external:
                G.add_edge(ex, central_prev)

            node_offset += len(G_prev)

        # Update external for next level
        external = new_external
        # Central remains the same
        central = central_prev

    return G, central, external

# Visualize hierarchical network (small version: level 3 recommended for clarity)
G_vis, _, _ = generate_hierarchical(2)  # smaller graph for visualization

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G_vis, seed=42)  # force-directed layout
nx.draw(
    G_vis,
    pos,
    node_size=50,
    node_color='skyblue',
    with_labels=False,
    edge_color='gray',
    alpha=0.7
)
plt.title("Hierarchical Network Visualization (Level 3)")
plt.show()
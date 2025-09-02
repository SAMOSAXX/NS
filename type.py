# Same size & avg degree as dataset
n = G.number_of_nodes()
avg_k = np.mean([deg for _, deg in G.degree()])
p = avg_k / (n-1)
k = int(round(avg_k))

if k % 2 == 1:
    k += 1  # WS requires even k

p_ws = 0.1  # rewiring probability
er_graph = generate_er(n, p)
sf_graph = generate_powerlaw_configuration(n, gamma=2.5)
ws_graph = generate_ws(n, k, p_ws)
full_report(er_graph, name="ER reference")
full_report(sf_graph, name="Scale-Free reference")
full_report(ws_graph, name="WS reference")
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

path_to_data = os.path.join(os.getcwd(), 'data', 'complejo')

# Conjunto de nodos
# nodo_id,x,y
N = pd.read_csv(os.path.join(path_to_data, 'nodos.csv'))

# Electrolineras
# nodo_id
E = pd.read_csv(os.path.join(path_to_data, 'electrolineras.csv'))
# GUARDAMOS UNA LISTA CON IDS DE ELECTROLINERAS
electrolineras = E['nodo_id'].values.tolist()

# Pasamos los nodos a un objeto networkx
G = nx.DiGraph()  # Usamos DiGraph para grafos dirigidos

for I in range(len(N)):
    node = N.iloc[I, 0]
    if node in electrolineras:
        G.add_node(node, pos=(N.iloc[I, 1], N.iloc[I, 2]), electrolinera=True)
    else:
        G.add_node(node, pos=(N.iloc[I, 1], N.iloc[I, 2]), electrolinera=False)

# Conjunto de aristas con información de distancia y tiempo
A = pd.read_csv(os.path.join(path_to_data, 'aristas.csv'))

# Pasamos las aristas a un objeto networkx
for I in range(len(A)):
    G.add_edge(A.iloc[I, 0], A.iloc[I, 1])

# electrolineras de color rojo
# nodos normales de color azul
pos = nx.get_node_attributes(G, 'pos')
node_colors = ['red' if G.nodes[node]['electrolinera'] else 'blue' for node in G.nodes]
edge_colors = ['black' if G.nodes[node]['electrolinera'] else 'blue' for node in G.nodes]

plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.patch.set_alpha(0)  # Fondo transparente

nx.draw(G, pos, with_labels=True, node_size=100, node_color=node_colors,
        edgecolors=edge_colors, arrows=True, arrowstyle='-|>', arrowsize=10)

plt.show()
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

path_to_data = 'simple'

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

for i in range(len(N)):
    node = N.iloc[i, 0]
    if node in electrolineras:
        G.add_node(node, pos=(N.iloc[i, 1], N.iloc[i, 2]), electrolinera=True)
    else:
        G.add_node(node, pos=(N.iloc[i, 1], N.iloc[i, 2]), electrolinera=False)

# Conjunto de aristas con información de distancia y tiempo
A = pd.read_csv(os.path.join(path_to_data, 'aristas.csv'))

# Pasamos las aristas a un objeto networkx
for i in range(len(A)):
    G.add_edge(A.iloc[i, 0], A.iloc[i, 1], tiempo=A.iloc[i, 2], energia=A.iloc[i, 3], temperatura=A.iloc[i, 4])

#electrolineras de color rojo
#nodos normales de color azul
pos = nx.get_node_attributes(G, 'pos')
colors = ['red' if G.nodes[node]['electrolinera'] else 'blue' for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_size=100, node_color=colors, arrows=True, arrowstyle='-|>', arrowsize=10)

plt.show()

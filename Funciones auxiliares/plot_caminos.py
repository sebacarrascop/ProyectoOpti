import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np
from matplotlib.lines import Line2D

# Leer los caminos desde el archivo resultados-final.txt
caminos = []
with open('resultados-final.txt') as f:
    for line in f:
        if line.startswith('1'):
            camino = [int(nodo) for nodo in line.strip().split(' -> ')]
            if camino not in caminos:
                caminos.append(camino)

# Cargar los datos de los nodos y electrolineras
path_to_data = os.path.join(os.getcwd(), 'data', 'LasCondes')

N = pd.read_csv(os.path.join(path_to_data, 'nodos.csv'))
E = pd.read_csv(os.path.join(path_to_data, 'electrolineras.csv'))

electrolineras = E['nodo_id'].values.tolist()

# Crear el grafo
G = nx.DiGraph()

for I in range(len(N)):
    node = N.iloc[I, 0]
    if node in electrolineras:
        G.add_node(node, pos=(N.iloc[I, 1], N.iloc[I, 2]), electrolinera=True)
    else:
        G.add_node(node, pos=(N.iloc[I, 1], N.iloc[I, 2]), electrolinera=False)

A = pd.read_csv(os.path.join(path_to_data, 'aristas.csv'))

for I in range(len(A)):
    G.add_edge(A.iloc[I, 0], A.iloc[I, 1])

# Dibujar el grafo con los colores de nodos y aristas
pos = nx.get_node_attributes(G, 'pos')
node_colors = ['green' if G.nodes[node]['electrolinera'] else 'gold' for node in G.nodes]
node_colors[0] = 'red'
node_colors[-1] = 'blue'

node_sizes = [100 if node == 1 or node == len(N) else 50 for node in G.nodes]

plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.patch.set_alpha(0)

nx.draw(G, pos, with_labels=False, node_color=node_colors,
        edgecolors='black', arrows=True, arrowstyle='-|>', arrowsize=5,
        width=1, edge_color='grey', alpha=0.7, node_size=node_sizes)

# Generar una lista de colores diferentes para los caminos
colors = [cm.tab10(i) for i in np.linspace(0, 1, len(caminos))]

# Dibujar los caminos con colores diferentes y crear leyendas
legend_elements = []
for i, camino in enumerate(caminos):
    path_edges = list(zip(camino[:-1], camino[1:]))
    color = colors[i]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=[color], width=2)
    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f'Camino {i+1}'))

# Añadir las leyendas para los nodos
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, label='Nodo normal'))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Electrolinera'))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Nodo inicial'))
legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Nodo final'))


# Añadir la leyenda al gráfico de manera horizontal
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)


ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

plt.show()

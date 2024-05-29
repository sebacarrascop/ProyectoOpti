import pandas as pd
import random
import math
import os
import networkx as nx
import numpy as np

def distancia_euclidiana(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Configuración de rutas
path_to_data = 'simple'
nodos_file = os.path.join(path_to_data, 'nodos.csv')
electrolineras_file = os.path.join(path_to_data, 'electrolineras.csv')
output_file = os.path.join(path_to_data, 'aristas.csv')

# Cargar los datos
nodos = pd.read_csv(nodos_file)
electrolineras = pd.read_csv(electrolineras_file)
nodos['es_electrolinera'] = nodos['nodo_id'].isin(electrolineras['nodo_id'])

# Número de nodos
num_nodos = len(nodos)

# Generar todas las posibles aristas con sus distancias
edges = []
nodo_ids = nodos['nodo_id'].tolist()
for i in range(num_nodos):
    for j in range(i + 1, num_nodos):
        nodo1 = nodo_ids[i]
        nodo2 = nodo_ids[j]
        if nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0] and nodos.loc[nodos['nodo_id'] == nodo2, 'es_electrolinera'].values[0]:
            continue  # No conectar dos electrolineras
        x1, y1 = nodos.loc[nodos['nodo_id'] == nodo1, ['x', 'y']].values[0]
        x2, y2 = nodos.loc[nodos['nodo_id'] == nodo2, ['x', 'y']].values[0]
        distancia = distancia_euclidiana(x1, y1, x2, y2)
        edges.append((distancia, nodo1, nodo2))

# Ordenar las aristas por distancia
edges.sort()

# Crear el árbol generador mínimo (MST) usando el algoritmo de Kruskal
aristas = []
parent = {}
rank = {}
for nodo in nodo_ids:
    parent[nodo] = nodo
    rank[nodo] = 0

def find(parent, i):
    if parent[i] == i:
        return i
    else:
        parent[i] = find(parent, parent[i])
        return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

num_aristas = 0
i = 0
while num_aristas < num_nodos - 1:
    distancia, nodo1, nodo2 = edges[i]
    i += 1
    x = find(parent, nodo1)
    y = find(parent, nodo2)
    if x != y:
        num_aristas += 1
        union(parent, rank, x, y)
        tiempo = distancia / 10
        es_electrolinera1 = nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0]
        es_electrolinera2 = nodos.loc[nodos['nodo_id'] == nodo2, 'es_electrolinera'].values[0]
        variacion_energia = -distancia / 2 if es_electrolinera1 else distancia / 2
        variacion_temperatura = -distancia / 10 if es_electrolinera1 else distancia / 10
        aristas.append([nodo1, nodo2, tiempo, variacion_energia, variacion_temperatura])

# Asegurar que el grafo sea fuertemente conexo añadiendo solo las aristas necesarias
def es_fuertemente_conexo(aristas, nodo_ids):
    G = nx.DiGraph()
    for arista in aristas:
        G.add_edge(arista[0], arista[1])
    return nx.is_strongly_connected(G)

# Añadir aristas adicionales para asegurar la fuerte conectividad
aristas_existentes = set((n1, n2) for n1, n2, _, _, _ in aristas)
while not es_fuertemente_conexo(aristas, nodo_ids):
    nodo1, nodo2 = random.sample(nodo_ids, 2)
    if (nodo1, nodo2) not in aristas_existentes:
        if nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0] and nodos.loc[nodos['nodo_id'] == nodo2, 'es_electrolinera'].values[0]:
            continue  # No conectar dos electrolineras
        x1, y1 = nodos.loc[nodos['nodo_id'] == nodo1, ['x', 'y']].values[0]
        x2, y2 = nodos.loc[nodos['nodo_id'] == nodo2, ['x', 'y']].values[0]
        distancia = distancia_euclidiana(x1, y1, x2, y2)
        tiempo = distancia / 10
        es_electrolinera1 = nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0]
        es_electrolinera2 = nodos.loc[nodos['nodo_id'] == nodo2, 'es_electrolinera'].values[0]
        variacion_energia = -distancia / 2 if es_electrolinera1 else distancia / 2
        variacion_temperatura = -distancia / 10 if es_electrolinera1 else distancia / 10
        aristas.append([nodo1, nodo2, tiempo, variacion_energia, variacion_temperatura])
        aristas_existentes.add((nodo1, nodo2))

# Convertir a DataFrame y guardar en CSV
aristas_df = pd.DataFrame(aristas, columns=['nodo1', 'nodo2', 'tiempo', 'variacion_energia', 'variacion_temperatura'])
aristas_df.to_csv(output_file, index=False)

print(f"Archivo '{output_file}' generado con éxito.")

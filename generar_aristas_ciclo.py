import pandas as pd
import random
import math
import os
import networkx as nx
import numpy as np

def distancia_manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

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

# Ordenar nodos por distancia, cosa que los mas cercanos esten juntos, excepto si son 2 electrolineras
nodos = nodos.sort_values(by=['x', 'y'])
nodo_ids = nodos['nodo_id'].tolist()
for i in range(num_nodos - 1):
    nodo1 = nodo_ids[i]
    nodo2 = nodo_ids[(i + 1)]
    if nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0] and nodos.loc[nodos['nodo_id'] == nodo2, 'es_electrolinera'].values[0]:
        nodo_ids[i], nodo_ids[i + 1] = nodo_ids[i + 1], nodo_ids[i]


# Primero generar un ciclo circular para asegurar que es conexo excepto la ultima arista
edges = []
nodo_ids = nodos['nodo_id'].tolist()
for i in range(num_nodos - 1):
    nodo1 = nodo_ids[i]
    nodo2 = nodo_ids[(i + 1)]
    x1, y1 = nodos.loc[nodos['nodo_id'] == nodo1, ['x', 'y']].values[0]
    x2, y2 = nodos.loc[nodos['nodo_id'] == nodo2, ['x', 'y']].values[0]
    distancia = distancia_manhattan(x1, y1, x2, y2)
    edges.append((distancia, nodo1, nodo2))

# Añadir más aristas random, un 10% de las posibles, no pueden ser entre electrolineras, ni salir de la ultima arista o llegar a la primera
for i in range(int(0.1 * num_nodos * (num_nodos - 1) / 2)):
    nodo1 = random.choice(nodo_ids)
    nodo2 = random.choice(nodo_ids)
    while nodo1 == nodo2 or nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0] and nodos.loc[nodos['nodo_id'] == nodo2, 'es_electrolinera'].values[0] or nodo1 == nodo_ids[-1] or nodo2 == nodo_ids[0] and distancia_manhattan(nodos.loc[nodos['nodo_id'] == nodo1, ['x', 'y']].values[0][0], nodos.loc[nodos['nodo_id'] == nodo1, ['x', 'y']].values[0][1], nodos.loc[nodos['nodo_id'] == nodo2, ['x', 'y']].values[0][0], nodos.loc[nodos['nodo_id'] == nodo2, ['x', 'y']].values[0][1]) > 4:
        nodo1 = random.choice(nodo_ids)
        nodo2 = random.choice(nodo_ids)
    x1, y1 = nodos.loc[nodos['nodo_id'] == nodo1, ['x', 'y']].values[0]
    x2, y2 = nodos.loc[nodos['nodo_id'] == nodo2, ['x', 'y']].values[0]
    distancia = distancia_manhattan(x1, y1, x2, y2)
    edges.append((distancia, nodo1, nodo2))

# nodo1,nodo2,tiempo,energia_requerida,variacion_temperatura

def tiempo(distancia):
    return distancia / 15

def energia_requerida(distancia):
    return distancia / 10

def variacion_temperatura(distancia, nodo1):
    if nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0]:
        return - distancia / 20
    return distancia / 20

# escribir las aristas en un archivo, añaadiendo la información de tiempo, energia_requerida y variación de temperatura
with open(output_file, 'w') as f:
    f.write('nodo1,nodo2,tiempo,energia_requerida,variacion_temperatura\n')
    for distancia, nodo1, nodo2 in edges:
        tiempo_ = tiempo(distancia)
        energia = energia_requerida(distancia)
        temperatura = variacion_temperatura(distancia, nodo1)
        f.write(f'{nodo1},{nodo2},{tiempo_},{energia},{temperatura}\n')
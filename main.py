import os
import gurobipy as gp
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


path_to_data = os.path.join(os.getcwd(), 'simple')

# Conjunto de nodos
# nodo_id,x,y
N = pd.read_csv(os.path.join(path_to_data, 'nodos.csv'))

# Electrolineras 
# nodo_id,potencia_de_carga,carga_por_unidad_tiempo
E = pd.read_csv(os.path.join(path_to_data, 'electrolineras.csv'))
# GUARDAMOS UNA LISTA CON IDS DE ELECTROLINERAS
electrolineras = E['nodo_id'].values.tolist()

# Pasamos los nodos a un objeto networkx
G = nx.DiGraph()  # Cambiado a DiGraph para aristas dirigidas

for i in range(len(N)):
    node = N.iloc[i, 0]
    if node in electrolineras:
        G.add_node(node, pos=(N.iloc[i, 1], N.iloc[i, 2]), electrolinera=True)
    else:
        G.add_node(node, pos=(N.iloc[i, 1], N.iloc[i, 2]), electrolinera=False)

# Conjunto de aristas con información de distancia y tiempo
# nodo1,nodo2,tiempo,variacion_energia,variacion_temperatura
A = pd.read_csv(os.path.join(path_to_data, 'aristas.csv'))

# Pasamos las aristas a un objeto networkx
for i in range(len(A)):
    G.add_edge(A.iloc[i, 0], A.iloc[i, 1], tiempo=A.iloc[i, 2], energia=A.iloc[i, 3], temperatura=A.iloc[i, 4])

# Definimos conjunto de tipos de auto V
# tipo,potencia_de_carga,temperatura_max,factor_consumo,tamano_bateria,temperatura_ideal_operacion
V = pd.read_csv(os.path.join(path_to_data, 'tipo_auto.csv'))

# Definimos conjunto de autos
#id,tipo,temperatura_inicial,porcentaje_inicial_bateria
C = pd.read_csv(os.path.join(path_to_data, 'autos.csv'))

# Obtenemos parametros
# porcentaje_max_bateria,porcentaje_min_bateria
parametros = pd.read_csv(os.path.join(path_to_data, 'parametros.csv'))
B_max = parametros.iloc[0, 0]
B_min = parametros.iloc[0, 1]

# imprimimos el grafo con matplotlib
# SIN ETIQUETAS
# LAS ARISTAS SON DIRIGIDAS
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=False, node_size=100, node_color='blue', arrows=True, arrowstyle='-|>', arrowsize=10)
plt.show()

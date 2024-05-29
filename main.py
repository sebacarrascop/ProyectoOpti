import os
import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


path_to_data = os.path.join(os.getcwd(), 'simple')

# Conjunto de nodos
# nodo_id,x,y
N = pd.read_csv(os.path.join(path_to_data, 'nodos.csv'))
cantidad_nodos = len(N)

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
cantidad_autos = len(C)

# Obtenemos parametros
# porcentaje_max_bateria,porcentaje_min_bateria
parametros = pd.read_csv(os.path.join(path_to_data, 'parametros.csv'))
B_max = parametros.iloc[0, 0]
B_min = parametros.iloc[0, 1]




### MODELO DE OPTIMIZACION ###

m = gp.Model("electric_cars")


## PARAMETROS ##

# Matriz de incidencia tiempo de viaje entre nodos (SUME 1 PARA MANTENER QUE EL INDICE DE LOS NODOS PARTE DE 1)
# por ende la primera fila y columna siempre seran 0.
T_ij = np.zeros((cantidad_nodos + 1, cantidad_nodos + 1))
for i, j, data in G.edges(data=True):
    T_ij[i, j] = data['tiempo']

## VARIABLES ## 

# Se define variable X_ij: 1 si el nodo i y el nodo j pertenecen al camino elegido del auto k
# Suponemos que cantidad_autos es el rango para k y G es tu grafo
X_ijk = m.addVars(G.edges(), range(cantidad_autos), vtype=gp.GRB.BINARY, name="X")

# Se define variable E_jk: energía porcentual de la batería del auto k al salir del nodo j.
E_jk = m.addVars(G.nodes(), range(cantidad_autos), vtype=gp.GRB.CONTINUOUS, name="E")

# Se define variable R_ik: tiempo de carga en el punto de carga i del auto k
R_ik = m.addVars(G.nodes(), range(cantidad_autos), vtype=gp.GRB.CONTINUOUS, name="R")

# Se define variable F_ik: temperatura de la batería del auto k en el nodo i
F_ik = m.addVars(G.nodes(), range(cantidad_autos), vtype=gp.GRB.CONTINUOUS, name="F")

# Se define la variable U_ik: 1 si la tempertura de la bateria del auto k supera la tempertura ideal de operación en el nodo i
U_ik = m.addVars(G.nodes(), range(cantidad_autos), vtype=gp.GRB.BINARY, name="U")


obj = gp.quicksum(gp.quicksum(gp.quicksum(gp.quicksum(T_ij[i, j] * X_ijk[i, j, k] for i, j in G.edges()) for k in range(cantidad_autos))))
m.setObjective(obj, gp.GRB.MINIMIZE)


# imprimimos el grafo con matplotlib
# SIN ETIQUETAS
# LAS ARISTAS SON DIRIGIDAS
# pos = nx.get_node_attributes(G, 'pos')
# nx.draw(G, pos, with_labels=False, node_size=100, node_color='blue', arrows=True, arrowstyle='-|>', arrowsize=10)
# plt.show()
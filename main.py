import os
import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import numpy as np
import networkx as nx


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
G = nx.DiGraph()

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
cantidad_de_tipos_de_autos = len(V)
tipos_de_autos = V['tipo'].values.tolist()

# Definimos conjunto de autos
#id,tipo,temperatura_inicial,porcentaje_inicial_bateria
C = pd.read_csv(os.path.join(path_to_data, 'autos.csv'))
cantidad_autos = len(C)

# Obtenemos parametros
# porcentaje_max_bateria,porcentaje_min_bateria
parametros = pd.read_csv(os.path.join(path_to_data, 'parametros.csv'))
B_max = parametros.iloc[0, 0]
B_min = parametros.iloc[0, 1]


## PARAMETROS ##

# Matriz de incidencia tiempo de viaje entre nodos (SUME 1 PARA MANTENER QUE EL INDICE DE LOS NODOS PARTE DE 1)
# por ende la primera fila y columna siempre seran 0.
T_ij = np.zeros((cantidad_nodos + 1, cantidad_nodos + 1))
for i, j, data in G.edges(data=True):
    T_ij[i, j] = data['tiempo']

# Matriz de tipo de auto x cantidad de autos, tiene un 1 si el auto k es de tipo v
Y_vk = np.zeros((cantidad_de_tipos_de_autos + 1, cantidad_autos + 1))
for k in range(cantidad_autos):
    for v in range(cantidad_de_tipos_de_autos):
        if C.iloc[k, 1] == V.iloc[v, 0]: 
            Y_vk[v+1, k+1] = 1

# Indicador de si el vehiculo k se puede cargar en la electrolinera i porque cumple 
# con que la potencia de carga del vehiculo es mayor o igual a la potencia de carga
# entregada en el nodo i.
Z_ik = np.zeros((cantidad_autos + 1, cantidad_nodos + 1))
for k in range(cantidad_autos):
    for i in range(1, cantidad_nodos + 1):
        if G.nodes[i]['electrolinera']:
            if V.iloc[C.iloc[k, 1]-1,1] >= E[E['nodo_id'] == i]['potencia_de_carga'].values[0]:
                Z_ik[k, i] = 1

# Constante de porcentaje de carga por unidad de tiempo, donde la posicion i,v es 
# la constante de carga de la electrolinera i para el tipo de auto v
g_iv = np.zeros((cantidad_nodos + 1, cantidad_de_tipos_de_autos + 1))
for nodo_electrolinera in electrolineras:
    for tipo_auto in tipos_de_autos:
        r_i = V['potencia_de_carga'][V['tipo'] == tipo_auto].values[0]
        C_v = V['tamano_bateria'][V['tipo'] == tipo_auto].values[0]
        constante_tiempo = 60 # [min/hora]
        g_iv[nodo_electrolinera, tipo_auto] = (r_i) / (C_v * constante_tiempo)

print(g_iv)

### MODELO DE OPTIMIZACION ###

m = gp.Model("electric_cars")


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

# Define the objective function
obj = gp.quicksum(T_ij[i, j] * X_ijk[i, j, k] + R_ik[i, k] for k in range(cantidad_autos) for i in range(cantidad_nodos) for j in range(cantidad_nodos))
# Set the objective to minimize
m.setObjective(obj, gp.GRB.MINIMIZE)

# Restricciones
# Restriccion 1: hay un camino solucion para cada auto
for k in range(cantidad_autos):
    for node in G.nodes():
        
        '''
        for succesor in G.successors(node):
            print(f"Node: {node}, Succesor: {succesor}")
        '''
        out_flow = quicksum(X_ijk[node, j, k] for j in G.successors(node) if (node, j) in G.edges())
        in_flow = quicksum(X_ijk[j, node, k] for j in G.predecessors(node) if (j, node) in G.edges())

        if node == 1:
            m.addConstr(out_flow - in_flow == 1, f"flow_cons_{node}_{k}")
        elif node == cantidad_nodos:
            m.addConstr(out_flow - in_flow == -1, f"flow_cons_{node}_{k}")
        else:
            m.addConstr(out_flow - in_flow == 0, f"flow_cons_{node}_{k}")

# Restriccion 2: Energia
# Energia inicial es equivalente al porcentaje inicial de la bateria del auto

# Falta recorrer sobre I_j
for j in G.nodes():
    for k in range(cantidad_autos):
        if j == 1:
            m.addConstr(E_jk[j, k] == C.iloc[k, 3], name=f"Energia_inicial_auto_{k}")
        else:
            tipo_auto = C.iloc[k, 1]
            m.addConstr(E_jk[j, k] == E_jk[i,k] - G.edges[i, j]['energia'] * Y_vk[i, j, k] + R_ik[i, k] * g_iv[i, tipo_auto] * Z_ik[i, k] - U_ik (e_ij[i, j]/5), name=f"Energia_nodo_{j}_auto_{k}")


# Restriccion 3: Se relaciona la variable E_jk con la X_ijk

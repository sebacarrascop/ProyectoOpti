import os
import gurobipy as gp
from gurobipy import quicksum
import pandas as pd
import numpy as np
import networkx as nx

# Variables globales
BIG_M = 1e100
LITLLE_M = 1e-100
path_to_data = os.path.join(os.getcwd(), 'simple')

# Conjunto de nodos
# nodo_id,x,y
N = pd.read_csv(os.path.join(path_to_data, 'nodos.csv'))
cantidad_nodos = len(N)

# Electrolineras
# nodo_id,potencia_de_carga,carga_por_unidad_tiempo
E = pd.read_csv(os.path.join(path_to_data, 'electrolineras.csv'))
electrolineras = E['nodo_id'].values.tolist()

# Pasamos los nodos a un objeto networkx
G = nx.DiGraph()

for i in range(len(N)):
    node = N.iloc[i, 0]
    if node in electrolineras:
        G.add_node(node, pos=(N.iloc[i, 1], N.iloc[i, 2]), electrolinera=True)
    else:
        G.add_node(node, pos=(N.iloc[i, 1], N.iloc[i, 2]), electrolinera=False)

# Conjunto de aristas
# nodo1,nodo2,tiempo,variacion_energia,variacion_temperatura
A = pd.read_csv(os.path.join(path_to_data, 'aristas.csv'))

# Pasamos las aristas a un objeto networkx
for i in range(len(A)):
    G.add_edge(A.iloc[i, 0], A.iloc[i, 1], tiempo=A.iloc[i, 2],
               energia=A.iloc[i, 3], temperatura=A.iloc[i, 4])

# Definimos conjunto de tipos de auto V
# tipo,potencia_de_carga,temperatura_max,factor_consumo,tamano_bateria,temperatura_ideal_operacion
V = pd.read_csv(os.path.join(path_to_data, 'tipo_auto.csv'))
cantidad_de_tipos_de_autos = len(V)
tipos_de_autos = V['tipo'].values.tolist()

# Definimos conjunto de autos
# id,tipo,temperatura_inicial,porcentaje_inicial_bateria
C = pd.read_csv(os.path.join(path_to_data, 'autos.csv'))
cantidad_autos = len(C)

# Obtenemos parametros
# porcentaje_max_bateria,porcentaje_min_bateria
parametros = pd.read_csv(os.path.join(path_to_data, 'parametros.csv'))
B_max = parametros.iloc[0, 0]
B_min = parametros.iloc[0, 1]
T_max = parametros.iloc[0, 2]


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
Z_ik = np.zeros((cantidad_nodos + 1, cantidad_autos + 1))
for k in range(cantidad_autos):
    for i in range(1, cantidad_nodos + 1):
        if G.nodes[i]['electrolinera']:
            if V.iloc[C.iloc[k, 1]-1, 1] >= E[E['nodo_id'] == i]['potencia_de_carga'].values[0]:
                Z_ik[i, k+1] = 1

# Constante de porcentaje de carga por unidad de tiempo, donde la posicion i,v es
# la constante de carga de la electrolinera i para el tipo de auto v
g_iv = np.zeros((cantidad_nodos + 1, cantidad_de_tipos_de_autos + 1))
for nodo_electrolinera in electrolineras:
    for tipo_auto in tipos_de_autos:
        r_i = V['potencia_de_carga'][V['tipo'] == tipo_auto].values[0]
        C_v = V['tamano_bateria'][V['tipo'] == tipo_auto].values[0]
        constante_tiempo = 60  # [min/hora]
        g_iv[nodo_electrolinera, tipo_auto] = (r_i) / (C_v * constante_tiempo)

e_ij = np.zeros((cantidad_nodos + 1, cantidad_nodos + 1))
for i, j, data in G.edges(data=True):
    e_ij[i, j] = data['energia']

# Matriz de variación de temperatura entre nodos
w_ij = np.zeros((cantidad_nodos + 1, cantidad_nodos + 1))
for i, j, data in G.edges(data=True):
    w_ij[i, j] = A["variacion_temperatura"][A["nodo1"]
                                            == i][A["nodo2"] == j].values[0]

### MODELO DE OPTIMIZACION ###

m = gp.Model("electric_cars")


## VARIABLES ##

# Se define variable X_ij: 1 si el nodo i y el nodo j pertenecen al camino elegido del auto k
# Suponemos que cantidad_autos es el rango para k y G es tu grafo
X_ijk = m.addVars(G.edges(), range(cantidad_autos),
                 vtype=gp.GRB.BINARY, name="X")

# Se define variable E_jk: energía porcentual de la batería del auto k al salir del nodo j.
E_jk = m.addVars(G.nodes(), range(cantidad_autos),
                 vtype=gp.GRB.CONTINUOUS, name="E")

# Se define variable R_ik: tiempo de carga en el punto de carga i del auto k
R_ik = m.addVars(G.nodes(), range(cantidad_autos),
                 vtype=gp.GRB.CONTINUOUS, name="R")

# Se define variable F_ik: temperatura de la batería del auto k en el nodo i
F_ik = m.addVars(G.nodes(), range(cantidad_autos),
                 vtype=gp.GRB.CONTINUOUS, name="F")

# Se define la variable U_ik: 1 si la tempertura de la bateria del auto k supera la tempertura ideal de operación en el nodo i
U_ik = m.addVars(G.nodes(), range(cantidad_autos),
                 vtype=gp.GRB.BINARY, name="U")

# Define the objective function
obj = gp.quicksum(T_ij[i, j] * X_ijk[i, j, k] + R_ik[i, k]
                  for k in range(cantidad_autos) for i, j in G.edges())
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
        out_flow = quicksum(X_ijk[node, j, k]
                            for j in G.successors(node) if (node, j) in G.edges())
        in_flow = quicksum(X_ijk[j, node, k] for j in G.predecessors(
            node) if (j, node) in G.edges())

        if node == 1:
            m.addConstr(out_flow - in_flow == 1, f"flow_cons_{node}_{k}")
        elif node == cantidad_nodos:
            m.addConstr(out_flow - in_flow == -1, f"flow_cons_{node}_{k}")
        else:
            m.addConstr(out_flow - in_flow == 0, f"flow_cons_{node}_{k}")

# Restriccion 2: Energia
# Energia inicial es equivalente al porcentaje inicial de la bateria del auto

# for i, j in G.edges():
#     for k in range(cantidad_autos):
#         if i == 1:
#             m.addConstr(E_jk[i, k] == C.iloc[k, 3],name=f"Energia_inicial_auto_{k}")
#         else:
#             tipo_auto = C.iloc[k, 1]  
#             factor_consumo = V[V['tipo'] == (tipo_auto)]['factor_consumo'].values[0]
#             print(f"Auto: {k+1} Arista: ({i}, {j}) tipo_auto: {tipo_auto} e_ij: {e_ij[i, j]} g_iv: {g_iv[i, tipo_auto]}")
#             # m.addConstr(E_jk[j, k] == E_jk[i, k] - e_ij[i, j] * factor_consumo * Y_vk[tipo_auto, k + 1]+ R_ik[i, k]
#             #             * g_iv[i, tipo_auto] * Z_ik[i, k+1] - U_ik[i, k] * e_ij[i, j] / 5, name=f"Energia_nodo_{j}_auto_{k}")
#             m.addConstr(E_jk[j, k] == E_jk[i, k] - e_ij[i, j])

for k in range(cantidad_autos):
    E_0k = C.iloc[k, 3]
    tipo_auto = C.iloc[k, 1]
    factor_consumo = V[V['tipo'] == (tipo_auto)]['factor_consumo'].values[0]
    sum = quicksum(E_0k - factor_consumo * e_ij[i, j] * X_ijk[i, j, k] for i, j in G.edges())
    m.addConstr(E_jk[j, k] == sum, name=f"Energia_inicial_auto_{k}")
    
    # m.addConstr(E_jk[j, k] == E_jk[i, k] - e_ij[i, j]
    #             * X_ijk[i, j, k] * factor_consumo)
    #             * g_iv[i, tipo_auto] * Z_ik[i, k+1] - U_ik[i, k] * e_ij[i, j] / 5, name=f"Energia_nodo_{j}_auto_{k}"))
    # m.addConstr(E_jk[j, k] == E_jk[i, k] - e_ij[i, j] * factor_consumo * Y_vk[tipo_auto, k + 1]+ R_ik[i, k]
    #             * g_iv[i, tipo_auto] * Z_ik[i, k+1] - U_ik[i, k] * e_ij[i, j] / 5, name=f"Energia_nodo_{j}_auto_{k}")


# # Restriccion 3: Se relaciona la variable E_jk con la X_ijk que cuando no escoja el camino la energía
# # gastada es cero y cuando se elige tomar a un valor maximo teórico
# for k in range(cantidad_autos):
#     for i, j in G.edges():
#         m.addConstr(E_jk[j, k] <= BIG_M * X_ijk[i, j, k],
#                     name=f"Energia_{j}_{k}_1")

# # Restriccion 4: Se relaciona la variable Rki con la Xkij de manera que cuando no escoja el camino el
# # tiempo de carga es cero y cuando se elige el camino sumado a que pertenece a una electrolinera
# # tomará un valor m´aximo te´orico, utilizando el m´etodo de la Big M (valor M muy grande).
# for k in range(cantidad_autos):
#     for i, j in G.edges():
#         if G.nodes[i]['electrolinera']:
#             m.addConstr(R_ik[i, k] <= BIG_M *
#                         X_ijk[i, j, k], name=f"R_{i}_{k}")
#         else:
#             m.addConstr(R_ik[i, k] == 0, name=f"R_{i}_{k}")


# # 5. Se limita la carga de la batería para aumentar la vida util de la batería.
# for k in range(cantidad_autos):
#     for i in G.nodes():
#         m.addConstr(E_jk[i, k] <= B_max, name=f"Max_Bateria_{i}_{k}")
#         m.addConstr(E_jk[i, k] >= B_min, name=f"Min_Bateria_{i}_{k}")

# # 6. Se define que solo se puede cargar en electrolineras y que el tiempo total de recarga no excederá
# # jamás el tiempo que tomaría cargar la batería por completo.
# for i in G.nodes():
#     for k in range(cantidad_autos):
#         for v in range(1, cantidad_de_tipos_de_autos + 1):
#             m.addConstr(g_iv[i, v] * R_ik[i, k] <= B_max *
#                         Z_ik[i, k], name=f"Tiempo_Carga_{i}_{k}")


# # 7. Primero1,  se limita la  + 1temperafor i, j in G.edges():or, pero para la cota inferior.
# for i in G.nodes():
#     for k in range(cantidad_autos):
#         for v in range(1, cantidad_de_tipos_de_autos + 1):
#             m.addConstr(F_ik[i, k] <= V.iloc[v-1, 5]*Y_vk[v, k] +
#                         BIG_M * U_ik[i, k], name=f"Temperatura_{i}_{k}_sup")


# # 8. Siguiendo la misma idea anterior, pero para la cota inferior
# for i in G.nodes():
#     for k in range(cantidad_autos):
#         for v in range(1, cantidad_de_tipos_de_autos + 1):
#             m.addConstr(F_ik[i, k] >= V.iloc[v-1, 5]*Y_vk[v, k] +
#                         LITLLE_M - BIG_M * (1 - U_ik[i, k]), name=f"Temperatura_sub_{i}_{k}_inf")

# # 9. Se relaciona las variables Uk de manera que el valor m ́aximo que puede tomar es cuando Xij toma el valor de uno,
# # pues solo si es parte del camino elegido Uk puede tomar valor uno.
# for k in range(cantidad_autos):
#     for i, j in G.edges():
#         m.addConstr(U_ik[i, k] <= X_ijk[i, j, k], name=f"U_{i}_{k}")


# # 11. Se relaciona la variable Fik con la Xijk de manera que cuando no escoja el camino la temperatura
# # no cambia y cuando se elige tomar a un valor maximo teórico
# for k in range(cantidad_autos):
#     for i, j in G.edges():
#         m.addConstr(F_ik[j, k] <= T_max * X_ijk[i, j, k],
#                     name=f"Temperatura_max_{j}_{k}")

# # 10. Se define el aumento y disminucion de temperatura
# for k in range(cantidad_autos):
#     for i in G.nodes():
#         if i == 1:
#             m.addConstr(F_ik[i, k] == C.iloc[k, 2],
#                         name=f"Temperatura_inicial_{i}_{k}")
#             continue
#         for j in G.successors(i):
#             m.addConstr(F_ik[j, k] == F_ik[i, k] + w_ij[i, j] * X_ijk[i, j, k],
#                         name=f"Temperatura_{j}_{k}")

# Optimize model
m.optimize()

# Print the objective value
print(f"Objetivo: {m.objVal}")

for k in range(cantidad_autos):
    print(f"\nCAMINO AUTO {k+1}:")
    camino = []
    nodo_actual = 1  # Nodo inicial
    while nodo_actual != cantidad_nodos:  # Nodo final
        camino.append(nodo_actual)
        for i, j in G.edges():
            if X_ijk[i, j, k].x > 0.5 and i == nodo_actual:
                nodo_actual = j
                break
    camino.append(cantidad_nodos)
    print(" -> ".join(map(str, camino)))
    energia_final = E_jk[cantidad_nodos, k].x
    print(f"Energía final en el nodo {cantidad_nodos}: {energia_final:.2f}%")
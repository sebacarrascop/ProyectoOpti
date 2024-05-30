import pandas as pd
import random
import os

# VARIABLES GLOABLES
PORCENTAJE_ARISTAS_EXTRA = 0.1
DISTACIA_MAXIMA = 4

# Calcular tiempo, energía requerida y variación de temperatura
def tiempo(distancia):
    return distancia / 15

def energia_requerida(distancia):
    return distancia / 10

def variacion_temperatura(distancia, nodo1):
    if nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0]:
        return - distancia / 20
    return distancia / 20

# Definir la distancia de Manhattan
def distancia_manhattan(node1, node2):
    x1, y1 = nodos.loc[nodos['nodo_id'] == node1, ['x', 'y']].values[0]
    x2, y2 = nodos.loc[nodos['nodo_id'] == node2, ['x', 'y']].values[0]
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

# Ordenar los nodos por distancia Manhattan desde el nodo 1
nodos['distancia_a_1'] = nodos['nodo_id'].apply(lambda x: distancia_manhattan(1, x))
nodos = nodos.sort_values(by='distancia_a_1')

nodo_ids = nodos['nodo_id'].tolist()

# Asegurar que no hay dos electrolineras seguidas
def evitar_electrolineras_consecutivas(nodo_ids, nodos):
    for i in range(0, len(nodo_ids) - 1):
        if nodos.loc[nodos['nodo_id'] == nodo_ids[i], 'es_electrolinera'].values[0] and \
           nodos.loc[nodos['nodo_id'] == nodo_ids[i + 1], 'es_electrolinera'].values[0]:
            print(f'Intercambiando nodos {nodo_ids[i]} y {nodo_ids[i + 1]}')
            # Buscar el próximo nodo que no sea electrolinera para hacer swap
            for j in range(i + 2, len(nodo_ids)):
                if not nodos.loc[nodos['nodo_id'] == nodo_ids[j], 'es_electrolinera'].values[0]:
                    nodo_ids[i + 1], nodo_ids[j] = nodo_ids[j], nodo_ids[i + 1]
                    break
        
    return nodo_ids

nodo_ids = evitar_electrolineras_consecutivas(nodo_ids, nodos)

# Generar un ciclo circular para asegurar la conexión
edges = []
for i in range(len(nodo_ids) - 1):
    nodo1 = nodo_ids[i]
    nodo2 = nodo_ids[i + 1]
    distancia = distancia_manhattan(nodo1, nodo2)
    edges.append((nodo1, nodo2, distancia))

# Añadir más aristas aleatorias según las condiciones
num_extra_edges = int(PORCENTAJE_ARISTAS_EXTRA * len(nodo_ids) * (len(nodo_ids) - 1) / 2)
for _ in range(num_extra_edges):
    nodo1 = random.choice(nodo_ids)
    nodo2 = random.choice(nodo_ids)
    while (nodo1 == nodo2 or
           (nodos.loc[nodos['nodo_id'] == nodo1, 'es_electrolinera'].values[0] and
            nodos.loc[nodos['nodo_id'] == nodo2, 'es_electrolinera'].values[0]) or
           nodo1 == 1 or
           nodo2 == nodo_ids[-1] or
           distancia_manhattan(nodo1, nodo2) > DISTACIA_MAXIMA):
        nodo1 = random.choice(nodo_ids)
        nodo2 = random.choice(nodo_ids)
    distancia = distancia_manhattan(nodo1, nodo2)
    edges.append((nodo1, nodo2, distancia))

# Escribir las aristas en el archivo de salida
with open(output_file, 'w') as f:
    f.write('nodo1,nodo2,tiempo,energia_requerida,variacion_temperatura\n')
    for nodo1, nodo2, distancia in edges:
        tiempo_ = tiempo(distancia)
        energia = energia_requerida(distancia)
        temperatura = variacion_temperatura(distancia, nodo1)
        f.write(f'{nodo1},{nodo2},{tiempo_},{energia},{temperatura}\n')

import pandas as pd
import random
import os

# Definir la ruta al directorio de datos
path_to_data = os.path.join(os.getcwd(), 'data', 'LasCondes')

# Cargar los nodos y las electrolineras
nodos = pd.read_csv(os.path.join(path_to_data, 'nodos.csv'))
electrolineras = pd.read_csv(os.path.join(path_to_data, 'electrolineras.csv'))
nodos['es_electrolinera'] = nodos['nodo_id'].isin(electrolineras['nodo_id'])

# Lista para almacenar las aristas
aristas = []

# Identificar el nodo inicial y final
nodo_inicial = nodos['nodo_id'].min()  # el nodo 1
nodo_final = nodos['nodo_id'].max()

# Elegir un número aleatorio de nodos destino desde el nodo inicial (sin incluir el último)
num_nodos_destino = random.randint(2, (len(nodos) - 1)//16)
destinos_iniciales = set(nodos[nodos['nodo_id'] != nodo_final]['nodo_id'].sample(num_nodos_destino, replace=False))

for nodo in destinos_iniciales:
    # Conectar el nodo inicial con varios destinos sin formar autociclos
    if nodo != nodo_inicial:
        aristas.append([nodo_inicial, nodo])

destinos_restantes = set(nodos['nodo_id']) - destinos_iniciales - {nodo_inicial}

nodos_sin_salida = set(nodos['nodo_id'])

# Asegurar la conexión entre los nodos evitando autociclos
while destinos_restantes:
    for arista in aristas:
        nodos_sin_salida.discard(arista[0])
    nodo_destino = random.choice(list(destinos_restantes))
    # Elegir un nodo de origen que no sea el mismo que el destino
    posibles_origenes = list(nodos_sin_salida - {nodo_destino})
    if posibles_origenes:
        nodo_origen = random.choice(posibles_origenes)
        aristas.append([nodo_origen, nodo_destino])
        destinos_restantes.discard(nodo_destino)

# Actualizar nodos sin salida después de conectar los nodos
for arista in aristas:
    nodos_sin_salida.discard(arista[0])  

# Conectar nodos sin salida con nodo final evitando autociclos
for nodo in nodos_sin_salida:
    if nodo != nodo_final:
        aristas.append([nodo, nodo_final])

# Convertir la lista de aristas a DataFrame
aristas_df = pd.DataFrame(aristas, columns=['nodo1', 'nodo2'])

# Calcular la distancia de Manhattan entre los nodos
def distancia_manhattan(nodo1, nodo2):
    x1, y1 = nodos.loc[nodos['nodo_id'] == nodo1, ['x', 'y']].values[0]
    x2, y2 = nodos.loc[nodos['nodo_id'] == nodo2, ['x', 'y']].values[0]
    return abs(x1 - x2) + abs(y1 - y2)

aristas_df['distancia'] = aristas_df.apply(lambda x: distancia_manhattan(x['nodo1'], x['nodo2']), axis=1)

# Calcular el tiempo, energía requerida y variación de temperatura
aristas_df['tiempo'] = aristas_df['distancia'] / 8
# Que sea random en cada entrada
for i in range(len(aristas_df)):
    aristas_df.loc[i, 'energia_requerida'] = 0.05 + random.uniform(-0.04, 0.04)
aristas_df['variacion_temperatura'] = aristas_df.apply(lambda x: -1 if x['nodo1'] in electrolineras['nodo_id'].values else 1, axis=1)

# quitar columnas de distancia
aristas_df = aristas_df.drop(columns=['distancia'])

# Guardar el DataFrame como CSV si es necesario
aristas_df.to_csv(os.path.join(path_to_data, 'aristas.csv'), index=False)

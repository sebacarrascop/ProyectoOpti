import pandas as pd
import numpy as np

# Número total de nodos requeridos
total_nodos = 150

# random sample of id from 1 to 1000
new_node_ids = np.random.choice(range(1, 1001), total_nodos, replace=False)

# Escribir los datos en un archivo CSV con otra columna potencia_de_carga con valor 22 fijo
new_nodes = pd.DataFrame({
    "nodo_id": new_node_ids,
    "potencia_de_carga": 22
})

new_nodes.to_csv('electrolineras.csv', index=False)
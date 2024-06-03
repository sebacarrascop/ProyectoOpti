import pandas as pd
import numpy as np

# Número total de nodos requeridos
total_nodos = 100


# Generar datos para los nodos faltantes
new_node_ids = np.arange(1, total_nodos + 1)
new_x = np.random.uniform(low=0.0, high=1000, size=(total_nodos))
new_y = np.random.uniform(low=0.0, high=1000, size=(total_nodos))

# Crear DataFrame para los nuevos nodos
new_nodes = pd.DataFrame({
    "nodo_id": new_node_ids,
    "x": new_x,
    "y": new_y
})


# Asegúrate de que los datos están en formato correcto y ordenados por nodo_id
new_nodes.sort_values(by='nodo_id', inplace=True)

# Mostrar o guardar los datos
print(new_nodes.head(60))  # Imprime los primeros 60 para verificar
new_nodes.to_csv('nodos.csv', index=False)  # Opcional: guardar en un archivo CSV

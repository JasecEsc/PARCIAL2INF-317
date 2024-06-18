##demil
import numpy as np
from scipy.sparse import csr_matrix

# Parámetros de las matrices
rows = 10
cols = 10
density = 0.2  # Proporción de elementos no cero

# Generar matrices densas aleatorias con una cierta densidad de elementos no cero
dense_matrix1 = np.random.choice([0, 1], size=(rows, cols), p=[1-density, density]) * np.random.randint(1, 10, size=(rows, cols))
dense_matrix2 = np.random.choice([0, 1], size=(rows, cols), p=[1-density, density]) * np.random.randint(1, 10, size=(rows, cols))

# Convertir las matrices densas a formato disperso (CSR)
sparse_matrix1 = csr_matrix(dense_matrix1)
sparse_matrix2 = csr_matrix(dense_matrix2)

# Multiplicar las matrices dispersas
result_sparse = sparse_matrix1.dot(sparse_matrix2)

# Convertir el resultado a una matriz densa para visualizar (opcional)
result_dense = result_sparse.todense()

print("Matriz Densa 1:")
print(dense_matrix1)

print("\nMatriz Dispersa 1 (CSR):")
print(sparse_matrix1)

print("\nMatriz Densa 2:")
print(dense_matrix2)

print("\nMatriz Dispersa 2 (CSR):")
print(sparse_matrix2)

print("\nResultado de la multiplicación (Matriz Dispersa):")
print(result_sparse)

print("\nResultado de la multiplicación (Matriz Densa):")
print(result_dense)
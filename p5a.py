import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, vstack
from multiprocessing import Pool, cpu_count

# Parámetros de las matrices
rows = 1000
cols = 1000
density = 0.2  # Proporción de elementos no cero

# Generar matrices densas aleatorias con una cierta densidad de elementos no cero
dense_matrix1 = np.random.choice([0, 1], size=(rows, cols), p=[1-density, density]) * np.random.randint(1, 10, size=(rows, cols))
dense_matrix2 = np.random.choice([0, 1], size=(rows, cols), p=[1-density, density]) * np.random.randint(1, 10, size=(rows, cols))

# Convertir las matrices densas a formato disperso (CSR)
sparse_matrix1 = csr_matrix(dense_matrix1)
sparse_matrix2 = csr_matrix(dense_matrix2)

# Determinar el número de procesos a utilizar (por defecto, el número de núcleos de CPU)
num_processes = cpu_count()
print(f"Número de procesos a utilizar: {num_processes}")

# Multiplicación paralela por filas usando multiprocessing.Pool y cpu_count
def multiply_row(row_index):
    row_result = sparse_matrix1.getrow(row_index).dot(sparse_matrix2)
    return row_result

if __name__ == "__main__":
    # Crear un pool de procesos
    with Pool(processes=num_processes) as pool:
        # Aplicar la función multiply_row a cada índice de fila en paralelo
        result_parallel = pool.map(multiply_row, range(rows))

    # Concatenar resultados en una matriz dispersa COO
    result_coo_rows = coo_matrix((0, cols))  # Inicializar matriz dispersa COO vacía
    for result in result_parallel:
        result_coo_rows = vstack([result_coo_rows, result])

    # Convertir el resultado a una matriz densa para visualizar (opcional)
    result_dense_parallel = result_coo_rows.todense()

    # Imprimir resultados
    print("Matriz Densa 1:")
    print(dense_matrix1)

    print("\nMatriz Dispersa 1 (CSR):")
    print(sparse_matrix1)

    print("\nMatriz Densa 2:")
    print(dense_matrix2)

    print("\nMatriz Dispersa 2 (CSR):")
    print(sparse_matrix2)

    print("\nResultado de la multiplicación paralela (Matriz Dispersa):")
    print(result_coo_rows)

    print("\nResultado de la multiplicación paralela (Matriz Densa):")
    print(result_dense_parallel)
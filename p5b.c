#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 200
#define COLS 200
#define DENSITY 0.2

// Estructura para matriz dispersa
typedef struct {
    int row;
    int col;
    double value;
} SparseMatrixEntry;

// Estructura para nodo en una lista enlazada de matriz dispersa
typedef struct Node {
    SparseMatrixEntry entry;
    struct Node *next;
} Node;

// Función para agregar un nodo a la lista enlazada
void addNode(Node **head, int row, int col, double value) {
    Node *newNode = (Node *)malloc(sizeof(Node));
    newNode->entry.row = row;
    newNode->entry.col = col;
    newNode->entry.value = value;
    newNode->next = *head;
    *head = newNode;
}

// Función para multiplicar matrices dispersas por filas usando OpenMP
void multiply_sparse_matrices(Node *matrix1, Node *matrix2, Node **result) {
    #pragma omp parallel for
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            double sum = 0.0;
            Node *rowNode = matrix1;
            while (rowNode != NULL) {
                if (rowNode->entry.row == i) {
                    Node *colNode = matrix2;
                    while (colNode != NULL) {
                        if (colNode->entry.col == j && rowNode->entry.col == colNode->entry.row) {
                            sum += rowNode->entry.value * colNode->entry.value;
                        }
                        colNode = colNode->next;
                    }
                }
                rowNode = rowNode->next;
            }
            if (sum != 0.0) {
                #pragma omp critical
                {
                    addNode(result, i, j, sum);
                }
            }
        }
    }
}

int main() {
    Node *matrix1 = NULL;
    Node *matrix2 = NULL;
    Node *result = NULL;

    // Inicializar matrices dispersas con valores aleatorios según la densidad
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if ((double) rand() / RAND_MAX < DENSITY) {
                double value1 = (double) rand() / RAND_MAX * 10.0; // Valor entre 0 y 10
                double value2 = (double) rand() / RAND_MAX * 10.0; // Valor entre 0 y 10
                addNode(&matrix1, i, j, value1);
                addNode(&matrix2, i, j, value2);
            }
        }
    }

    // Multiplicar matrices dispersas por filas usando OpenMP
    multiply_sparse_matrices(matrix1, matrix2, &result);

    // Imprimir resultado (opcional)
    printf("Resultado de la multiplicación:\n");
    Node *current = result;
    while (current != NULL) {
        printf("(%d, %d): %.2f\n", current->entry.row, current->entry.col, current->entry.value);
        current = current->next;
    }

    // Liberar memoria
    Node *temp;
    while (matrix1 != NULL) {
        temp = matrix1;
        matrix1 = matrix1->next;
        free(temp);
    }
    while (matrix2 != NULL) {
        temp = matrix2;
        matrix2 = matrix2->next;
        free(temp);
    }
    while (result != NULL) {
        temp = result;
        result = result->next;
        free(temp);
    }

    return 0;
}
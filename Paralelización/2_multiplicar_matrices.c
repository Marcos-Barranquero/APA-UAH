#include <stdio.h>
#include <stdlib.h>

int numero_aleatorio()
{
    // Genero un número aleatorio entre 0 y 50:
    return rand() % 27;
}


int **crear_matriz(int filas, int columnas)
{
    /** Crea una matriz de filas y columnas con memoria dinámica **/
    int **matriz = (int **) malloc(filas * sizeof(int *));
    for (int i = 0; i < filas; ++i)
    {
        matriz[i] = (int *) malloc(columnas * sizeof(int));
    }


    return matriz;

}



int **rellenar_matriz(int **matriz, int filas, int columnas)
{
    /** Rellena una matriz con números aleatorios entre 0 y 50. **/
    for (int i = 0; i < filas; ++i)
    {
        for (int j = 0; j < columnas; ++j)
        {
            matriz[i][j] = numero_aleatorio();
        }
    }

    return matriz;

}

void imprimir_matriz(int **matriz, int filas, int columnas)
{
    /** Imprime una matriz por terminal. **/
    printf("\n");

    for (int i = 0; i < filas; ++i)
    {
        printf("M[%i]:{", i);
        for (int j = 0; j < columnas; ++j)
        {
            int numero = matriz[i][j];
            printf("%i, ", numero);
        }
        printf("}\n");

    }
}

int main(int argc, const char *argv[])
{

    // Variable para determinar el tamaño de las matrices:
    int filas_primera_matriz = 3;
    int columnas_primera_matriz = 3;

    int filas_segunda_matriz = 3;
    int columnas_segunda_matriz = 3;

    int filas_matriz_resultado = filas_primera_matriz;
    int columnas_matriz_resultado = columnas_segunda_matriz;

    // Creación de matrices:
    int **primera_matriz = crear_matriz(filas_primera_matriz, columnas_primera_matriz);
    int **segunda_matriz = crear_matriz(filas_segunda_matriz, columnas_segunda_matriz);
    int **matriz_resultado = crear_matriz(filas_matriz_resultado, columnas_matriz_resultado);


    if (primera_matriz == NULL || segunda_matriz == NULL || matriz_resultado == NULL)
        // Si hay errores muestro msj. de error y detengo la ejecución:
    {
        printf("Error en la asignación de memoria.  ");

    } else
        // Si no, relleno matrices y después multiplico:
    {
        // Relleno matrices con números aleatorios:
        primera_matriz = rellenar_matriz(primera_matriz, filas_primera_matriz, columnas_primera_matriz);
        segunda_matriz = rellenar_matriz(segunda_matriz, filas_segunda_matriz, columnas_segunda_matriz);

        // Imprimo las matrices:
        printf("Primera matriz: ");
        imprimir_matriz(primera_matriz, filas_primera_matriz, columnas_primera_matriz);
        printf("Segunda matriz: ");
        imprimir_matriz(segunda_matriz, filas_segunda_matriz, columnas_segunda_matriz);
        printf("Matriz resultado: ");
        imprimir_matriz(matriz_resultado, filas_segunda_matriz, columnas_segunda_matriz);



        // Las multiplico (filas x columnas):
        for (int i = 0; i < filas_matriz_resultado; ++i)
        {
            for (int j = 0; j < columnas_matriz_resultado; ++j)
            {
                int resultado_temporal = 0;
                for (int k = 0; k < columnas_primera_matriz; ++k)
                {
                    int n1 = primera_matriz[i][k];
                    int n2 = segunda_matriz[k][j];
                    resultado_temporal = resultado_temporal + (n1 * n2);
                    
                }
                matriz_resultado[i][j] = resultado_temporal;
            }
        }

        // Imprimo la matriz resultado:
        printf("Matriz resultado de M1 * M2: ");
        imprimir_matriz(matriz_resultado, filas_matriz_resultado, columnas_matriz_resultado);



    }


    return 0;
}



#include <stdio.h>
#include <stdlib.h>

int numero_aleatorio() {
    // Genero un número aleatorio entre 0 y 50:
    return rand() % 50;
}

int main() {

    // Variable para determinar el tamaño de los vectores:
    int tamano = 5;

    // Declaración y asignación de memoria en vectores:
    int *primer_vector = (int *) malloc(tamano * sizeof(int));
    int *segundo_vector = (int *) malloc(tamano * sizeof(int));

    int *vector_resultado = (int *) malloc(tamano * sizeof(int));


    if (primer_vector == NULL || segundo_vector == NULL)
        // Si hay errores muestro msj. de error y detengo la ejecución:
    {
        printf("Error en la asignación de memoria.  ");

    } else
        // Si no, relleno vectores y después sumo.
    {
        // Relleno vectores con números aleatorios:
        for (int i = 0; i < tamano; ++i) {
            *(primer_vector + i) = numero_aleatorio();
            *(segundo_vector + i) = numero_aleatorio();
        }


        // Imprimo los dos vectores:
        printf("\n");
        printf("V1: ");
        for (int i = 0; i < tamano; ++i) {
            printf("%i, ", *(primer_vector + i));
        }

        printf("\n");
        printf("V2: ");

        for (int i = 0; i < tamano; ++i) {
            printf("%i, ", *(segundo_vector + i));
        }

        printf("\n");

        // Los sumo:
        for (int k = 0; k < tamano; ++k) {
            *(vector_resultado + k) = *(primer_vector + k) + *(segundo_vector + k);
        }

        printf("V1 + V2: ");
        // Imprimo la suma:
        for (int l = 0; l < tamano; ++l) {
            printf("%i, ", *(vector_resultado + l));
        }

    }


    return 0;
}



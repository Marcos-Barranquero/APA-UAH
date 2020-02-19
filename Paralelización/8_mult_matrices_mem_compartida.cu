/*
Marcos Barranquero Fernández - 21/02/2019
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define FILAS 16
#define COLUMNAS 16
#define BYTES_MATRIZ (FILAS * COLUMNAS * sizeof(int))
#define ANCHO_SUBMATRIZ 4
#define ANCHO_MATRIZ 16


__global__ void kernel_mult_mem_compartida(int* d_m1, int* d_m2, int* d_mr) {
	// Declaro las submatrices de memoria compartida:
	__shared__ int s_m1[ANCHO_SUBMATRIZ][ANCHO_SUBMATRIZ];
	__shared__ int s_m2[ANCHO_SUBMATRIZ][ANCHO_SUBMATRIZ];

   // Obtengo variables de hilo y bloque para posicionarme:
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Obtengo fila y columna sobre la que estoy trabajando:
	int fila = bx * ANCHO_SUBMATRIZ + ty;
	int columna = by * ANCHO_SUBMATRIZ + tx;

	// Variable que almacenará el valor de la multiplicación de filas * columnas
	int valor_acumulado = 0;


	for (int i = 0; i < ANCHO_MATRIZ / ANCHO_SUBMATRIZ; ++i) {

		// Cargo las matrices en la memoria compartida:
		s_m1[threadIdx.y][threadIdx.x] = d_m1[fila*ANCHO_MATRIZ + (i*ANCHO_SUBMATRIZ + tx)];
		s_m2[threadIdx.y][threadIdx.x] = d_m2[(i*ANCHO_SUBMATRIZ + ty)*ANCHO_MATRIZ + columna];

		__syncthreads();
		
		// Realizo la multiplicación, acelerada por leer de la memoria compartida:
		for (int k = 0; k < ANCHO_SUBMATRIZ; ++k)
		{
			valor_acumulado += s_m1[ty][k] * s_m2[k][tx];
		}

		__syncthreads();

	}
	// Cargo resultado en matriz resultado en memoria del DEVICE:
	d_mr[fila*ANCHO_MATRIZ + columna] = valor_acumulado;
}



void multiplicarMatrices(int *h_m1, int *h_m2, int *h_mr)
{
	// Punteros a matrices en DEVICE:
	int *d_m1;
	int *d_m2;
	int *d_mr;

	// Reservo memoria en DEVICE:
	cudaMalloc((void **)&d_m1, BYTES_MATRIZ);
	cudaMalloc((void **)&d_m2, BYTES_MATRIZ);
	cudaMalloc((void **)&d_mr, BYTES_MATRIZ);

	// Muevo de HOST a DEVICE:
	cudaMemcpy(d_m1, h_m1, BYTES_MATRIZ, cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2, h_m2, BYTES_MATRIZ, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mr, h_mr, BYTES_MATRIZ, cudaMemcpyHostToDevice);

	// Defino tamaño de bloques:
	dim3 matriz_bloques(ANCHO_SUBMATRIZ, ANCHO_SUBMATRIZ);
	dim3 matriz_hilos(ANCHO_MATRIZ/ANCHO_SUBMATRIZ, ANCHO_MATRIZ / ANCHO_SUBMATRIZ);

	kernel_multiplicar <<< matriz_bloques, matriz_hilos >>> (d_m1, d_m2, d_mr);

	// Espero a que termine de operar:
	cudaDeviceSynchronize();

	// Devolvemos resultado de DEVICE a HOST:
	cudaMemcpy(h_mr, d_mr, BYTES_MATRIZ, cudaMemcpyDeviceToHost);

	// Libero memoria de DEVICE:
	cudaFree(d_m1);
	cudaFree(d_m2);
	cudaFree(d_mr);


}


void rellenarMatriz(int *h_m, int filas, int columnas)
{
	/* Rellena una matriz de filasxcolumnas con números aleatorios.
	*/
	srand(time(NULL));
	for (int i = 0; i < filas; ++i) {
		for (int j = 0; j < columnas; ++j) {
			*(h_m + i * columnas + j) = rand() % 101;
		}
	}
}

void pintarMatriz(int *h_m, int filas, int columnas) {
	/*
	* Imprime matriz por pantalla.
	*/
	for (int i = 0; i < columnas; i++) {
		printf("[");
		for (int j = 0; j < filas; j++) {
			if (j != filas && j != 0) {
				printf("\t");
			}
			printf("%d", *(h_m + i * columnas + j));
		}
		printf("]\n");
	}
}

int main()
{

	// Declaración de matrices en host:
	int* h_m1 = (int *)malloc(BYTES_MATRIZ);
	int* h_m2 = (int *)malloc(BYTES_MATRIZ);
	int* h_mr = (int *)malloc(BYTES_MATRIZ); // Matriz resultado.

	// Relleno con datos aleatorios las matrices:
	rellenarMatriz(h_m1, FILAS, COLUMNAS);
	rellenarMatriz(h_m2, FILAS, COLUMNAS);

	// Imprimo:
	printf("Matriz 1: \n");
	pintarMatriz(h_m1, FILAS, COLUMNAS);
	printf("Matriz 2: \n");
	pintarMatriz(h_m2, FILAS, COLUMNAS);

	// Multiplico:
	multiplicarMatrices(h_m1, h_m2, h_mr);

	// Imprimo resultado:
	printf("Matriz resultado: \n");
	pintarMatriz(h_mr, FILAS, COLUMNAS);

	// Libero espacio en memoria:
	free(h_m1);
	free(h_m2);
	free(h_mr);

	return 0;


}


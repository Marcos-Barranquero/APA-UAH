
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void suma_vectores_cubo(int *d_v1, int *d_v2, int *d_vr)
{
	int id_vector = blockIdx.x * 8 + threadIdx.x;
	printf("Id: %d\n", id_vector);
	d_vr[id_vector] = d_v1[id_vector] + d_v2[id_vector];
}

int main()
{
	// Variables de tamaños
	const int ARRAY_SIZE = 24;
	const int ARRAY_BYTES = 24 * sizeof(int);

	// Declaro vectores y vector resultado
	int h_v1[ARRAY_SIZE];
	int h_v2[ARRAY_SIZE];
	int h_vr[ARRAY_SIZE];

	// Relleno v1 y v2:
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		h_v1[i] = i;
		h_v2[i] = i;
		h_vr[i] = 0;
	}

	// Declaro punteros a memoria GPU
	int * d_v1;
	int * d_v2;
	int * d_vr;

	// Asigno memoria GPU
	cudaMalloc((void**)&d_v1, ARRAY_BYTES);
	cudaMalloc((void**)&d_v2, ARRAY_BYTES);
	cudaMalloc((void**)&d_vr, ARRAY_BYTES);

	// Transfiero los arrays a la GPU:
	cudaMemcpy(d_v1, h_v1, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v2, h_v2, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vr, h_vr, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// Lanzo el kernel
	suma_vectores_cubo <<< 3, 8 >>> (d_v1, d_v2, d_vr);

	// Copio el resultado al HOST:
	cudaMemcpy(h_vr, d_vr, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		printf("V[%d]: %d\n", i, h_vr[i]);
	}










    return 0;
}


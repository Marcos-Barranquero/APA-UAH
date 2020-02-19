
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>

#define TRAPECIOS 6500
#define HILOS_POR_BLOQUE 256
#define BLOQUES_POR_GRID min(32, (TRAPECIOS+HILOS_POR_BLOQUE - 1) / HILOS_POR_BLOQUE)

using namespace::std;

__device__ float d_integral(float x)
{
	return cos(1 / x)*pow(x, 2) * (pow(x, 3) + 9) / (x + 3);
}

float h_integral(float x)
{
	return cos(1 / x)*pow(x, 2) * (pow(x, 3) + 9) / (x + 3);
}

__global__ void trapecios(float a, float b, float h, float * resultado)
{
	__shared__ float resultados_parciales[HILOS_POR_BLOQUE];

	// Nº iteración
	int iteracion = threadIdx.x + blockIdx.x * blockDim.x;

	// Variable que guarda el resultado:
	float resultado_parcial = 0;

	// Mientras queden trapecios por calcular:
	while (iteracion < TRAPECIOS)
	{
		// Calculo integral
		if (iteracion != 0)
		{
			resultado_parcial += d_integral(a + h * iteracion);
		}

		// Paso al siguiente
		iteracion += blockDim.x *gridDim.x;

	}

	// Lo almaceno en el vector de resultados:
	resultados_parciales[threadIdx.x] = resultado_parcial;

	// Fase de reducción (algoritmo 2 de suma de vectores)
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (threadIdx.x < i)
		{
			// Sumo cada elemento con el elemento de al lado 
			resultados_parciales[threadIdx.x] += resultados_parciales[threadIdx.x + i];

		}
		// Reduzco el vector a la mitad
		i /= 2;
	}

	if (threadIdx.x == 0)
	{
		resultado[blockIdx.x] = resultados_parciales[0];
	}

}



int main()
{
	// Constantes de la integral
	const float a = 1.5f;
	const float b = 2.78f;
	float h = (b - a) / TRAPECIOS;

	// Variables de HOST:
	float *h_resultado = (float*)malloc(BLOQUES_POR_GRID * sizeof(float));
	float *d_resultado;

	// Reservo memoria en DEVICE:
	cudaMalloc(&d_resultado, BLOQUES_POR_GRID * sizeof(float));

	// Llamo al KERNEL:
	trapecios << <BLOQUES_POR_GRID, HILOS_POR_BLOQUE >> > (a, b, h, d_resultado);

	// Copio de vuelta el resultado a HOST:
	cudaMemcpy(h_resultado, d_resultado, BLOQUES_POR_GRID * sizeof(float), cudaMemcpyDeviceToHost);

	// Sumo los parciales:
	float suma_parciales = (h_integral(a) + h_integral(b)) / 2.0f;

	for (int i = 0; i < BLOQUES_POR_GRID; i++)
	{
		suma_parciales += h_resultado[i];
	}

	suma_parciales *= h;

	std::cout << "Resultado de integral con GPU: " << suma_parciales << std::endl;

	suma_parciales = (h_integral(a) + h_integral(b)) / 2.0f;
	for (int i = 1; i < TRAPECIOS; i++)
	{
		suma_parciales += h_integral(a + i * h);
	}

	suma_parciales *= h;

	std::cout << "Resultado de integral con CPU: " << suma_parciales << std::endl;

	cudaFree(d_resultado);
	free(h_resultado);
	return 0;

}

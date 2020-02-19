
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel_sumar(int *a, int *b, int *c, int *resultado)
/*
Guarda en resultado la suma de a + b + c.
*/
{
	*resultado = *a + *b + *c;
}


void sumar_en_cuda(int a, int b, int c, int* resultado)
{
	// Variables de la gráfica:
	int *dev_a;
	int *dev_b;
	int *dev_c;
	// Variable resultado:
	int *dev_resultado;

	// Reservo memoria en DEVICE para los 3 ints. Nota:(void **) es un parseo de puntero. 
	cudaMalloc((void **)&dev_a, sizeof(int));
	cudaMalloc((void **)&dev_b, sizeof(int));
	cudaMalloc((void **)&dev_c, sizeof(int));
	cudaMalloc((void **)&dev_resultado, sizeof(int));

	// Copio contenido del HOST al DEVICE: (No hace falta copiar resultado, pues no tiene aún valor).
	cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, &c, sizeof(int), cudaMemcpyHostToDevice);

	// Lanzo el kernel:
	kernel_sumar <<<100, 100 >>> (dev_a, dev_b, dev_c, dev_resultado);

	// Espero a que el kernel termine su ejecución:
	cudaDeviceSynchronize();

	// Copio de DEVICE a HOST: (guardo en c).
	cudaMemcpy(resultado, dev_resultado, sizeof(int), cudaMemcpyDeviceToHost);


	// Libero memoria del DEVICE:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_resultado);

}

int main()
{
	// Variable donde se almacenará el resultado.
	int resultado;

	// Llamo a la función de suma:
	sumar_en_cuda(3, 5, 8, &resultado);

	// Imprimo resultado:
	printf("El resultado es %d", resultado);
	return 0;
}



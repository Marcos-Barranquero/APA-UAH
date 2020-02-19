
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{

	// Marco la GPU como GPU a utilizar:
	cudaSetDevice(0);

	// Variable de las propiedades:
	cudaDeviceProp propiedades;

	// Obtengo propiedades de la GPU 0:
	cudaGetDeviceProperties(&propiedades,0);

	printf("Nombre de la GPU: %s\n", propiedades.name);
	// Multiplico por 2 debido a que es DDR y realiza operaciones por flanco de subida y bajada: 
	printf("Frecuencia de la memoria (GHz): %f\n", 2.0 * (propiedades.memoryClockRate / 1.0e6));

	printf("Interfaz de memoria (bits): %d\n", propiedades.memoryBusWidth);

	printf("Ancho de banda (GB/s): %f\n", 2.0*propiedades.memoryClockRate*(propiedades.memoryBusWidth / 8) / 1.0e6);



	return 0;
}

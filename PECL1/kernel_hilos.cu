
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <Windows.h>
using namespace std;

#define OBJETIVO 2048

#define DESP_POS 1
#define DESP_NEG -1
#define NO_DESP 0

#ifdef __INTELLISENSE__
void __syncthreads(); // Para evitar el error que da el intellisense con syncthreads y atomicadd
void atomicAdd(int *a, int b);
#endif

// Variables globales para recoger por parámetros
struct dimensionesMatriz {
	int numFilas;
	int numColumnas;
} dimMatriz;


dim3 dimGrid;	// Grid de bloques
dim3 dimBlock;	// Hilos por bloque

// Variables de control

// Juego automático o manual
bool automatico;
// Nº de bytes que ocupa la matriz
int bytesMatriz;
// Dificultad del juego
bool modoDiablo;

// Control de vidas:
int vidas;

// Funciones de juego
__host__ void juegoAutomatico(int *h_matriz, int *h_matrizResultado, int *d_matrizEntrada, int *d_matrizSalida);
__host__ void juegoManual(int *h_matriz, int *h_matrizResultado, int *d_matrizEntrada, int *d_matrizSalida);

// Funciones auxiliares en Device
__device__ int getElemento(int *matriz, int fila, int columna, dimensionesMatriz* d_dimMatriz);
__device__ void setElemento(int *matriz, int fila, int columna, int elemento, dimensionesMatriz* d_dimMatriz);

// Kernels 
	// Kernel movimiento
__global__ void kernelDesplazar(int *h_matrizEntrada, int *h_matrizSalida, int *despVertical, int *despHorizontal, dimensionesMatriz* d_dimMatriz);

	// Kernels auxiliares
__global__ void kernelSuma(int *h_matrizEntrada, int *h_matrizSalida, int *d_puntuacion, int *despVertical, int *despHorizontal, dimensionesMatriz* d_dimMatriz);
__global__ void kernelCopiarMatriz(int *d_matrizCopia, int *d_matrizPega, dimensionesMatriz* d_dimMatriz);
__global__ void kernelSetMatrizCeros(int *matriz, dimensionesMatriz* d_dimMatriz);

	// Funciones auxiliares de comprobación de estado de la matriz
__global__ void kernelComprobarIguales(int *d_matrizUno, int *d_matrizDos, bool* d_sonIguales, dimensionesMatriz* d_dimMatriz);
__global__ void kernelComprobarLlena(int *d_matrizUno, bool* d_estaLlena, dimensionesMatriz* d_dimMatriz);
__global__ void kernelComprobarMovimientosPosibles(int *d_matriz, bool *seguirJugando, dimensionesMatriz* d_dimMatriz);
__global__ void kernelComprobarSiHaGanado(int *d_matriz, bool* d_haGanado, dimensionesMatriz* d_dimMatriz);

// Funciones auxiliares en Host
__host__ void caracteristicasTarjeta();
__host__ void leerParametros(int argc, const char* argv[]);

	// Operaciones con matrices
__host__ void inicializarMatriz(int *h_matriz);
__host__ void rellenarMatrizconcero(int *h_matriz);
__host__ void pintarMatriz(int *h_matriz);
__host__ void copiarMatriz(int *h_matrizCopia, int *h_matrizPega);
__host__ void setElementoHost(int *h_matriz, int fila, int columna, int elemento);
__host__ void nuevaSemilla(int *h_matriz, int numSemillas);

	// Comprobadores
__host__ bool estaLlena(int* d_matriz);
__host__ bool finJuego(int* d_matriz);
__host__ bool movimientosPosibles(int* d_matriz);



	// Funciones de host de carga y guardado de matrices:
__host__ void escribirMatriz(int* h_matriz, string nombreJugador, int* puntuacion, int* movimientos);
__host__ bool leerMatriz(int* h_matriz, string nombreJugador, int* puntuacion, int* movimientos);

	// Funcion de movimiento en Host
__host__ bool desplazarMatriz(int *h_matrizEntrada, int *h_matrizSalida, int *d_matrizEntrada, int *d_matrizSalida, int *h_puntuacion, int despVertical, int despHorizontal);

// MAIN

int main(int argc, const char* argv[])
{
	leerParametros(argc, argv);

	// Declaracion de matrices en host:
	int* h_matriz = (int *)malloc(bytesMatriz);
	int* h_matrizResultado = (int *)malloc(bytesMatriz);

	// Punteros a matrices en DEVICE:
	int *d_matrizEntrada;
	int *d_matrizSalida;

	// Reserva de memoria en DEVICE
	cudaMalloc((void **)&d_matrizEntrada, bytesMatriz);
	cudaMalloc((void **)&d_matrizSalida, bytesMatriz);

	// Relleno las matrices con 0s:
	rellenarMatrizconcero(h_matriz);
	rellenarMatrizconcero(h_matrizResultado);

	if (automatico)
		juegoAutomatico(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida);
	else
		juegoManual(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida);

	// Libero la memoria de device
	cudaFree(d_matrizEntrada);
	cudaFree(d_matrizSalida);

	return 0;
}

// ----------- MODOS DE JUEGO ----------- //

__host__ void juegoAutomatico(int *h_matriz, int *h_matrizResultado, int *d_matrizEntrada, int *d_matrizSalida)
{
	cout << "+--------------------------------------------------------+" << endl;
	cout << "| Bienvenido al 16384, se ha elegido el modo automatico. |" << endl;
	cout << "+--------------------------------------------------------+" << endl;

	inicializarMatriz(h_matriz);

	// Se comprueban las caracteristicas de la tarjeta
	cout << "+--------------------------------------------------------+" << endl;
	caracteristicasTarjeta();
	cout << "+--------------------------------------------------------+" << endl;

	cout << endl;
	system("pause");
	system("cls");

	// Contador de movimientos
	int movimientos = 0;
	int puntuacion = 0;
	vidas = 5;

	// Variable control de entrada
	bool seguirJugando = false;
	bool ganado = false;

	while (!ganado && vidas > 0)
	{
		// Eligo un movimiento aleatorio
		int movimiento = rand() % 4;
		system("CLS");

		// Y lo hago
		switch (movimiento)
		{
			// PARAMETROS DESPLAZAR_MATRIZ -> matriz inicial, matriz resultado, desplazamiento eje y, desplazamiento eje x
		case 0:
			cout << "Muevo arriba " << endl;
			seguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, DESP_NEG, NO_DESP); // Desplazar arriba
			ganado = finJuego(d_matrizSalida);
			break;
		case 1:
			cout << "Muevo abajo " << endl;
			seguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, DESP_POS, NO_DESP); // Desplazar abajo
			ganado = finJuego(d_matrizSalida);
			break;
		case 2:
			cout << "Muevo izquierda " << endl;
			seguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, NO_DESP, DESP_NEG); // Desplazar izquierda
			ganado = finJuego(d_matrizSalida);
			break;
		case 3:
			cout << "Muevo derecha " << endl;
			seguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, NO_DESP, DESP_POS); // Desplazar derecha
			ganado = finJuego(d_matrizSalida);
			break;
		}

		movimientos++;

		copiarMatriz(h_matrizResultado, h_matriz);

		cout << "+------------------------------------------------------------+" << endl;
		printf("|Movimiento: %d\tPuntuacion: %d\t Vidas: %d                \n", movimientos, puntuacion, vidas);
		cout << "+------------------------------------------------------------+" << endl;
		pintarMatriz(h_matriz);

		if (!seguirJugando && vidas > 1)
		{
			cout << "+---------------------------------------------------------------------------------------------+" << endl;
			cout << "| No hay mas movimientos posibles, la maquina ha perdido. Hemos suspendido el test de Turing. |" << endl;
			cout << "+---------------------------------------------------------------------------------------------+" << endl;
			vidas -= 1;

			cout << endl;
			cout << "+---------------------------------------------------------------------------------------------+" << endl;
			cout << "| Lo intentamos de nuevo (si/no)?.                                                            |" << endl;
			cout << "+---------------------------------------------------------------------------------------------+" << endl;

			string otraVez;
			cin >> otraVez;

			if (otraVez == "no")
			{
				cout << "Hasta la vista, Baby. " << endl;
				exit(0);
			}
			else if(otraVez == "si")
			{
				rellenarMatrizconcero(h_matriz);
				rellenarMatrizconcero(h_matrizResultado);
				movimientos = 0;
				seguirJugando = true;
			}
		}
		else if (ganado)
		{
			cout << endl << "LA MAQUINA HA GANADO VIVA TURING " << endl;
			exit(0);
		}

		// Sleep chungo de C++. Cambiar el 100 por lo que se quiera
		//this_thread::sleep_for(chrono::milliseconds(100));

		// Si se quiere avanzar con enters descomentar esto:
		//system("PAUSE");
	}

	cout << "A la maquina no le quedan vidas. Fin de juego. Adios Terminator. " << endl;
	exit(0);


}

__host__ void juegoManual(int *h_matriz, int *h_matrizResultado, int *d_matrizEntrada, int *d_matrizSalida)
{
	// Muestro mensaje de bienvenida
	cout << "+----------------------------------------------------+" << endl;
	cout << "|Hola amigo bienvenido al 16384 que ganitas de jugar |" << endl;
	cout << "+----------------------------------------------------+" << endl;
	cout << endl;

	// Muestro características de la tarjeta
	cout << "+----------------------------------------------------+" << endl;
	caracteristicasTarjeta();
	cout << "+----------------------------------------------------+" << endl;

	// Variables de control y estados iniciales:
	int movimientos = 0; // Contador de movimientos por partida
	int puntuacion = 0; // Puntuación total
	vidas = 5; // Establezco vidas a 5. 

	char entrada1, entrada2; // Carácteres de lectura por teclado
	bool correcto = false; // Variable control de entrada

	bool puedeSeguirJugando = false; // Aún hay movimientos disponibles
	bool ganado = false; // Si ha ganado
	bool haGanadoYQuiereSeguir = false; // Comprobacion por si quiere seguir jugando despues de ganar

	// Recojo nombre de usuario
	string nombre;
	cout << "+----------------------------------------------------+" << endl;
	cout << "| Dame tu nombre amiguete:                           |" << endl;
	cout << "+----------------------------------------------------+" << endl;
	cin >> nombre;
	cout << endl;

	// Cargo (o no) la partida
	cout << "+----------------------------------------------------+" << endl;
	cout << "| Quieres cargar tu partida?                         |" << endl;
	cout << "+----------------------------------------------------+" << endl;
	string cargar;
	cin >> cargar;

	// Si quiere cargar y existe la partida, la cargo.
	if (cargar == "si" && leerMatriz(h_matriz, nombre, &movimientos, &puntuacion))
	{
		cout << "+----------------------------------------------------+" << endl;
		cout << "| Partida cargada.                                   |" << endl;
		cout << "+----------------------------------------------------+" << endl;
	}
	// Si no, establezco matriz.
	else
	{
		inicializarMatriz(h_matriz);
	}

	// Juego:

	while (true)
	{
		// Imprimo matriz y estadísticas
		system("CLS");
		cout << "+------------------------------------------------------------+" << endl;
		printf("|Movimiento: %d\tPuntuacion: %d\t Vidas: %d                \n", movimientos, puntuacion, vidas);
		cout << "+------------------------------------------------------------+" << endl;
		pintarMatriz(h_matriz);

		// Tengo que volver a comprobar la entrada.
		correcto = true;

		// Las teclas de movimiento hacen input de dos caracteres,
		// siendo el segundo el que nos importa para el movimiento
		entrada1 = getch();

		// Si el usuario quiere salir, se sale.
		if (entrada1 == 's')
			break;
		else
		{
			// Obtengo segundo caracter
			entrada2 = getch();

			// Realizo jugada:
			switch (entrada2)
			{
				// PARAMETROS DESPLAZAR_MATRIZ -> matriz inicial, matriz resultado, puntuacion, desplazamiento eje y, desplazamiento eje x
			case 72:
				puedeSeguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, DESP_NEG, NO_DESP); // Desplazar arriba
				ganado = finJuego(d_matrizSalida);
				break;
			case 80:
				puedeSeguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, DESP_POS, NO_DESP); // Desplazar abajo
				ganado = finJuego(d_matrizSalida);
				break;
			case 75:
				puedeSeguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, NO_DESP, DESP_NEG); // Desplazar izquierda
				ganado = finJuego(d_matrizSalida);
				break;
			case 77:
				puedeSeguirJugando = desplazarMatriz(h_matriz, h_matrizResultado, d_matrizEntrada, d_matrizSalida, &puntuacion, NO_DESP, DESP_POS); // Desplazar derecha
				ganado = finJuego(d_matrizSalida);
				break;
			default:
				cout << "Caracter incorrecto. " << endl;
				correcto = false;
			}
		}
		
		// Tras hacer la jugada, compruebo el estado de la matriz. 
		if (correcto)
		{
			// Copio resultado a matriz:
			copiarMatriz(h_matrizResultado, h_matriz);

			// Incremento movimientos
			movimientos++;

			// Si pierde y le quedan vidas y no estaba farmeando puntos.
			if (!puedeSeguirJugando && vidas > 1 && !haGanadoYQuiereSeguir)
			{
				// Resto una vida
				vidas -= 1;

				// Muestro mensaje por pantalla:
				cout << "+---------------------------------------------------------------------------+" << endl;
				cout << "| No hay mas movimientos posibles, fin de juego. Intentalo de nuevo.        |" << endl;
				cout << "| Te quedan: " << vidas << " vidas.                                         | " << endl;
				cout << "+---------------------------------------------------------------------------+" << endl;


				// Recojo si quiere seguir jugando:
				string otraVez;

				do
				{
					cout << "+---------------------------------------------------------------------------+" << endl;
					cout << "| Quieres intentarlo de nuevo (si/no)?                                      |" << endl;
					cout << "+---------------------------------------------------------------------------+" << endl;
					cin >> otraVez;
				}while (!(otraVez == "si") || !(otraVez != "no"));

				// Si no quiere seguir jugando, se sale.
				if (otraVez == "no")
				{
					cout << "Nos vemos amigo. " << endl;
					exit(0);
				}
				// Si se quiere seguir jugando, se resetean datos. 
				else
				{
					rellenarMatrizconcero(h_matriz);
					rellenarMatrizconcero(h_matrizResultado);
					movimientos = 0;
					ganado = false;
					haGanadoYQuiereSeguir = false;
					inicializarMatriz(h_matriz);
				}

			}
			// Si pierde y no le quedan vidas y no estaba farmeando puntos.
			else if (!puedeSeguirJugando && vidas == 1 && !haGanadoYQuiereSeguir)
			{
				vidas -= 1;
				cout << endl << "No hay mas movimientos posibles, fin del juego." << endl;
				cout << endl << "Además no te quedan vidas." << endl;
				cout << "Esta es tu puntuacion final: " << puntuacion << endl;
				exit(0);
			}
			// Si había ganado y ahora ya no puede seguir
			else if (!puedeSeguirJugando && haGanadoYQuiereSeguir)
			{
				// Muestro mensaje por pantalla:
				cout << "+---------------------------------------------------------------------------+" << endl;
				cout << endl << "| No hay mas movimientos posibles, fin de juego. Intentalo de nuevo." << endl;
				cout << endl << "| Te quedan: " << vidas << " vidas. " << endl;
				cout << "+----------------------------------------------------------------------------+" << endl;


				// Recojo si quiere seguir jugando:
				string otraVez;

				do
				{
					cout << "+---------------------------------------------------------------------------------------------+" << endl;
					cout << "| Quieres intentarlo de nuevo (si/no)?                                                        |" << endl;
					cout << "+---------------------------------------------------------------------------------------------+" << endl;
					cin >> otraVez;
				} while (otraVez != "si" || otraVez != "no");

				// Si no quiere seguir jugando, se sale.
				if (otraVez == "no")
				{
					cout << "Nos vemos amigo. " << endl;
					exit(0);
				}
				// Si se quiere seguir jugando, se resetean datos. 
				else
				{
					rellenarMatrizconcero(h_matriz);
					rellenarMatrizconcero(h_matrizResultado);
					movimientos = 0;
					ganado = false;
					haGanadoYQuiereSeguir = false;
					inicializarMatriz(h_matriz);
				}

			}
			// Si acaba de ganar
			else if (ganado && !haGanadoYQuiereSeguir)
			{
				cout << "+---------------------------------------------------------------------------+" << endl;
				cout << "| Felicidades campeon, has ganado. Esta es tu puntuacion final:        " << puntuacion << endl;
				cout << "+---------------------------------------------------------------------------+" << endl;

				string jugarMas;
				while (!(jugarMas == "si") && !(jugarMas == "no"))
				{
					cout << endl << "Quieres seguir jugando?" << endl;
					cin >> jugarMas;
				}

				if (jugarMas == "no")
				{
					cout << "Hasta luego!" << endl;
					exit(0);
				}
				else
				{
					haGanadoYQuiereSeguir = true;
				}
			}
		}
	}

	// Guardar partida
	cout << "Quieres guardar partida? " << endl;
	string entrada;
	cin >> entrada;

	if (entrada == "si")
	{
		escribirMatriz(h_matriz, nombre, &movimientos, &puntuacion);
		cout << "Matriz guardada con nombre: " + nombre << endl;
	}

}

// ----------- FUNCIONES DEVICE ----------- // 

__device__ int getElemento(int *d_matriz, int fila, int columna, dimensionesMatriz* d_dimMatriz)
/*
Dada una matriz, devuelve el elemento en [fila][columna]
*/
{
	return d_matriz[fila * d_dimMatriz->numColumnas + columna];
}

__device__ void setElemento(int *d_matriz, int fila, int columna, int elemento, dimensionesMatriz* d_dimMatriz)
/*
Dada una matriz, escribe el elemento en [fila][columna]
*/
{
	d_matriz[fila * d_dimMatriz->numColumnas + columna] = elemento;
}

// --------- KERNELS PRINCIPALES ----------- //

__global__ void kernelCopiarMatriz(int *d_matrizCopia, int *d_matrizPega, dimensionesMatriz* d_dimMatriz)
/*
Dada una matriz a copiar, se pega todo el contenido de esta en la matriz a pegar.
*/
{
	// Encuentro posicion:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	// Copio:
	int elemento_copiar = getElemento(d_matrizCopia, fila, columna, d_dimMatriz);
	// pego
	setElemento(d_matrizPega, fila, columna, elemento_copiar, d_dimMatriz);
}

__global__ void kernelSuma(int *d_matrizEntrada, int *d_matrizSalida, int *d_puntuacion, int *despVertical, int *despHorizontal, dimensionesMatriz* d_dimMatriz)
/*
Dada una matriz de entrada y una de salida, escribe las sumas por desplazamiento en la matriz de salida.
*/
{
	// Encuentro posicion:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	// Variables auxiliares para comprobaciones
	int ultimaPosicion, desplazamiento, posicionActual;
	bool esVertical;

	// Analizo que tipo de movimiento se esta haciendo
	if (*despVertical != 0)
	{
		// Si es vertical, ajusto parámetros:
		posicionActual = fila;
		desplazamiento = fila;
		esVertical = true;
		if (*despVertical == -1)
			ultimaPosicion = 0;
		else
			ultimaPosicion = d_dimMatriz->numFilas - 1;
	}
	else
	{
		// Si es horizontal, ajusto parámetros
		posicionActual = columna;
		desplazamiento = columna;
		esVertical = false;
		if (*despHorizontal == -1)
			ultimaPosicion = 0;
		else
			ultimaPosicion = d_dimMatriz->numColumnas - 1;
	}

	// Obtengo el elemento en la posicion
	int elemento = getElemento(d_matrizEntrada, fila, columna, d_dimMatriz);

	// Variable que controla si se multiplicare elm. x2 o no.
	bool multiplicarem = false;

	// Si no soy un 0:
	if (elemento != 0 && posicionActual != ultimaPosicion)
	{
		// Compruebo paridad de los elementos en la dirección en la que me desplazo. 
		int paridad = 1;

		// Casilla que compruebo en el bucle.
		int casilla;

		// Mientras no se encuentre un elemento distinto o se sobrepase la matriz
		do {
			// Casilla estudiada
			if (esVertical)
				casilla = getElemento(d_matrizEntrada, desplazamiento + *despVertical, columna, d_dimMatriz);
			else
				casilla = getElemento(d_matrizEntrada, fila, desplazamiento + *despHorizontal, d_dimMatriz);

			// Si es diferente al elemento y no es 0, rompemos el bucle. 
			if (casilla != elemento && casilla != 0) { break; }

			// Si hay otro elemento igual encima, aumento paridad
			if (casilla == elemento) { paridad += 1; }

			// Y sigo viendo
			desplazamiento += *despHorizontal + *despVertical;
		} while (desplazamiento != ultimaPosicion);

		// Si hay pares, pongo mult. a true.
		if (paridad % 2 == 0)
		{
			multiplicarem = true;
		}

		// Espero a todos los hilos
		__syncthreads();

		// Si debo multiplicar, multiplico
		if (multiplicarem)
		{
			// Encuentro la pos. del elemento a mul * 2 
			int casilla;
			desplazamiento = posicionActual; // Reseteamos el desplazamiento

			// Mientras haya 0s me desplazo.
			do {
				desplazamiento += *despHorizontal + *despVertical;
				if (esVertical)
					casilla = getElemento(d_matrizEntrada, desplazamiento, columna, d_dimMatriz);
				else
					casilla = getElemento(d_matrizEntrada, fila, desplazamiento, d_dimMatriz);
			} while (casilla != elemento);

			// Sumo la puntuacion parcial que ha obtenido cada hilo con una suma atomica
			atomicAdd(d_puntuacion, elemento * 2);

			// Duplico el elemento que tengo encima
			if (esVertical)
				setElemento(d_matrizSalida, desplazamiento, columna, elemento * 2, d_dimMatriz);
			else
				setElemento(d_matrizSalida, fila, desplazamiento, elemento * 2, d_dimMatriz);
		}
		// Si no, me escribo a mi mismo en la matriz de salida. 
		else
		{
			setElemento(d_matrizSalida, fila, columna, getElemento(d_matrizEntrada, fila, columna, d_dimMatriz), d_dimMatriz);
		}

		// Espero a que todos los hilos multipliquen.
		__syncthreads();
	}
	else
	{
		setElemento(d_matrizSalida, fila, columna, getElemento(d_matrizEntrada, fila, columna, d_dimMatriz), d_dimMatriz);
	}

	// Espero a que finalicen los hilos.
	__syncthreads();
}

__global__ void kernelSetMatrizCeros(int *matriz, dimensionesMatriz* d_dimMatriz)
/*
Dada una matriz, setea todas sus posiciones a 0.
*/
{
	// Encuentro posicion:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	// Elemento en la posici�n
	setElemento(matriz, fila, columna, 0, d_dimMatriz);

	// Espero a que el resto de hilos pongan 0s. 
	__syncthreads();
}

__global__ void kernelDesplazar(int *d_matrizEntrada, int *d_matrizSalida, int* despVertical, int* despHorizontal, dimensionesMatriz* d_dimMatriz)
/*
Dada una matriz, desplaza sus elementos 1 vez en la dirección indicada, si se puede.
*/
{
	// Encuentro posicion y elemento de mi bloque:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;
	int elemento = getElemento(d_matrizEntrada, fila, columna, d_dimMatriz);

	int ultimaPosicion, posicionActual;
	// Analizo que tipo de movimiento se esta haciendo
	if (*despVertical != 0)
	{
		posicionActual = fila;

		if (*despVertical == -1)
			ultimaPosicion = 0;
		else
			ultimaPosicion = d_dimMatriz->numFilas - 1;
	}
	else
	{
		posicionActual = columna;

		if (*despHorizontal == -1)
			ultimaPosicion = 0;
		else
			ultimaPosicion = d_dimMatriz->numColumnas - 1;
	}


	// Variable que dice si se debe mover o no.
	bool desplazarem = false;

	// Si soy distinto de 0 y no estoy en el limite
	if ((posicionActual != ultimaPosicion) && (elemento != 0))
	{
		// Si la casilla siguiente a la mía en el movimiento es un 0, desplazaré hacia esa dirección. 
		int casillaVecina = getElemento(d_matrizEntrada, fila + *despVertical, columna + *despHorizontal, d_dimMatriz);

		if (casillaVecina == 0)
		{
			desplazarem = true;
		}

		// Espero a que marquen el resto de hilos.
		__syncthreads();

		// Y desplazo:
		if (desplazarem)
		{
			//printf("Soy [%d][%d] (%d) y me desplazo. \n", fila, columna, elemento);
			setElemento(d_matrizSalida, fila + *despVertical, columna + *despHorizontal, elemento, d_dimMatriz);
		}
		// O escribo mi valor. 
		else
		{
			//printf("Soy [%d][%d] (%d) y NO me desplazo. \n", fila, columna, elemento);
			setElemento(d_matrizSalida, fila, columna, elemento, d_dimMatriz);
		}
		// Espero resto de hilos:
		__syncthreads();
	}
	// Si estoy en el limite
	else if (elemento != 0)
	{
		//printf("Soy [%d][%d] (%d) y NO me desplazo pq estoy al limite o soy un 0. \n", fila, columna, elemento);
		setElemento(d_matrizSalida, fila, columna, elemento, d_dimMatriz);
	}
	// Si no, soy un cero y no tengo que escribir nada porque d_matrizSalida es una matriz de 0s. 

	// Espero al resto de hilos
	__syncthreads();
}

// -------- KERNELS COMPROBADORES ---------- // 

__global__ void kernelComprobarIguales(int *d_matrizUno, int *d_matrizDos, bool* d_sonIguales, dimensionesMatriz* d_dimMatriz)
/*
Dadas dos matrices, deja sonIguales a true si lo son.
*/
{
	// Encuentro posicion:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;
	// Elemento min & mout:
	int elemento1 = getElemento(d_matrizUno, fila, columna, d_dimMatriz);
	int elemento2 = getElemento(d_matrizDos, fila, columna, d_dimMatriz);

	if (elemento1 != elemento2)
		*d_sonIguales = false;

	// Espero al resto de hilos:
	__syncthreads();
}

__global__ void kernelComprobarLlena(int *d_matriz, bool* d_estaLlena, dimensionesMatriz* d_dimMatriz)
/*
Dadas una matriz, pone estaLlena a false si hay algún 0 y, por tanto, no está llena.
*/
{
	// Encuentro posicion:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	// Elemento min & mout:
	int elemento = getElemento(d_matriz, fila, columna, d_dimMatriz);
	if (elemento == 0)
		*d_estaLlena = false;

	// Espero al resto de hilos:
	__syncthreads();
}

__global__ void kernelComprobarSiHaGanado(int *d_matriz, bool* d_haGanado, dimensionesMatriz* d_dimMatriz)
/*
Dadas una matriz, pone estaLlena a false si hay algún 0 y, por tanto, no está llena.
*/
{
	// Encuentro posicion:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	// Elemento min & mout:
	int elemento = getElemento(d_matriz, fila, columna, d_dimMatriz);
	if (elemento == OBJETIVO)
		*d_haGanado = true;

	// Espero al resto de hilos:
	__syncthreads();
}

__global__ void kernelComprobarMovimientosPosibles(int *d_matriz, bool *seguirJugando, dimensionesMatriz* d_dimMatriz)
/*
Comprueba si hay elementos posibles, si los hay, devuelve true. Si no hay movimientos posibles, devuelve false
*/
{
	// Encuentro posicion:
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	int elemento = getElemento(d_matriz, fila, columna, d_dimMatriz);

	bool seguirJugando_aux; // Booleano auxiliar para no escribir en el parametro directamente

	// Booleanos para ver donde en que direccion podemos movernos
	bool comprobarArr = true, comprobarAb = true, comprobarIzq = true, comprobarDer = true;
	// Booleanos para comprobar los elementos con los que no podemos combinarnos
	bool combinarArr = false, combinarAb = false, combinarIzq = false, combinarDer = false;

	// Comprobamos en que posicion estamos para no salirnos fuera de los rangos de la matriz
	if (fila == 0)
		comprobarArr = false;
	else if (fila == d_dimMatriz->numFilas - 1)
		comprobarAb = false;

	if (columna == 0)
		comprobarIzq = false;
	else if (columna == d_dimMatriz->numColumnas - 1)
		comprobarDer = false;

	int elementoEstudiado;

	if (comprobarArr) {
		elementoEstudiado = getElemento(d_matriz, fila - 1, columna, d_dimMatriz);
		if (elementoEstudiado == elemento)
			combinarArr = true;
	}
	if (comprobarAb) {
		elementoEstudiado = getElemento(d_matriz, fila + 1, columna, d_dimMatriz);
		if (elementoEstudiado == elemento)
			combinarAb = true;
	}
	if (comprobarDer) {
		elementoEstudiado = getElemento(d_matriz, fila, columna + 1, d_dimMatriz);
		if (elementoEstudiado == elemento)
			combinarDer = true;
	}
	if (comprobarIzq) {
		elementoEstudiado = getElemento(d_matriz, fila, columna - 1, d_dimMatriz);
		if (elementoEstudiado == elemento)
			combinarIzq = true;
	}

	seguirJugando_aux = combinarArr || combinarAb || combinarIzq || combinarDer;

	if (seguirJugando_aux)
		*seguirJugando = seguirJugando_aux;
}

// -------- FUNCIONES AUX HOST ----------- // 

__host__ void leerParametros(int argc, const char* argv[])
/*
Parsea los parámetros introducidos en la llamada al programa por consola, seteando
las variables del juego.
*/
{
	if ((argc != 5) || ((argv[1][0] != 'a') && (argv[1][0] != 'm')) || ((argv[2][0] != 'f') && (argv[2][0] != 'd')))
	{
		cout << "Error en la introduccion de parametros, los parametros son:\nautomatico/manual (a/m), facil/dificil (f/d), num_filas, num_columnas\n\nUso = nombreprograma a/m f/d num_filas num_columnas\n" << endl;
		exit(1);
	}
	else
	{
		dimMatriz.numFilas = atoi(argv[3]);
		dimMatriz.numColumnas = atoi(argv[4]);

		if (dimMatriz.numFilas != dimMatriz.numColumnas)
		{
			cout << "El numero de filas y de columnas no puede ser distinto, crack." << endl;
			exit(2);
		}

		bytesMatriz = atoi(argv[3]) * atoi(argv[4]) * sizeof(int);

		// Se dimensionan los hilos y los grids de bloques
		if (dimMatriz.numFilas % 2 == 0)
		{
			dim3 bloques(2, 2);
			dim3 hilos(dimMatriz.numFilas / 2, dimMatriz.numColumnas / 2);
			dimGrid = bloques;
			dimBlock = hilos;
		}
		else
		{
			dim3 bloques(1, 1);
			dim3 hilos(dimMatriz.numFilas, dimMatriz.numColumnas);
			dimGrid = bloques;
			dimBlock = hilos;
		}


		if (argv[1][0] == 'a')
			automatico = true;
		else
			automatico = false;

		if (argv[2][0] == 'f')
			modoDiablo = false;
		else
			modoDiablo = true;
	}
}

__host__ void pintarMatriz(int *h_matriz)
/*
Dada una matriz, la dibuja por pantalla.
*/
{
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	for (size_t i = 0; i < dimMatriz.numColumnas; i++)
	{
		SetConsoleTextAttribute(hConsole, 14);
		cout << ("+-------");

	}
	cout << "+" << endl;
	for (int i = 0; i < dimMatriz.numColumnas; i++)
	{
		for (int j = 0; j < dimMatriz.numFilas; j++)
		{
			// La funcion de print evalua en su interior si deberia poner un /t o no, dependiendo de la longitud del numero
			printf("[%d%s]", *(h_matriz + i * dimMatriz.numColumnas + j),
				*(h_matriz + i * dimMatriz.numColumnas + j) % 100000 == *(h_matriz + i * dimMatriz.numColumnas + j) ? "\t" : "");
		}
		printf("\n");
	}
	for (size_t i = 0; i < dimMatriz.numColumnas; i++)
	{
		cout << ("+-------");

	}
	cout << "+" << endl;

	SetConsoleTextAttribute(hConsole, 15);
}

__host__ void caracteristicasTarjeta()
/*
Saca por pantalla las caracteristicas de todas las tarjetas graficas del pc
*/
{
	// Recojo el número de tarjetas de la gráfica
	int numTarjetas;
	cudaGetDeviceCount(&numTarjetas);

	// Para cada una, imprimo sus características
	for (int i = 0; i < numTarjetas; i++) {
		cudaDeviceProp caracteristicas;
		cudaGetDeviceProperties(&caracteristicas, i);

		printf("Numero de dispositivo: %d\n", i);
		printf("  Nombre del dispositivo: %s\n", caracteristicas.name);
		printf("  Frecuencia del reloj de memoria (KHz): %d\n",
			caracteristicas.memoryClockRate);
		printf("  Interfaz de memoria (bits): %d\n",
			caracteristicas.memoryBusWidth);
		printf("  Ancho de banda de memoria (GB/s): %f\n",
			2.0*caracteristicas.memoryClockRate*(caracteristicas.memoryBusWidth / 8) / 1.0e6);
	}
}

// ------- OP. CON MATRIZ EN HOST ------- //

__host__ void inicializarMatriz(int *h_matriz)
/*
Dada una matriz, la rellena con 0s, 2s, 4s u 8s, aleatoriamente y dependiendo del nivel de dificultad elegido.
*/
{
	srand(time(NULL));

	// Contador de casillas rellenadas. Dependiendo de la dificultad, tiene un tope distinto. 
	int contadorSemillas = 0;
	int *posicionAleatoria;

	if (modoDiablo)
	{
		int array_posibles_numeros[] = { 2,4 };

		while ((contadorSemillas < 8) && (contadorSemillas < dimMatriz.numFilas * dimMatriz.numColumnas))	// Mientras no se hayan lanzado todas las semillas
		{
			posicionAleatoria = h_matriz + (rand() % dimMatriz.numColumnas) * dimMatriz.numColumnas + (rand() % dimMatriz.numFilas);	// Calculo una posicion aleatoria donde poner una de las semillas
			if (*posicionAleatoria == 0) // Si es 0 inicialmente, es decir, no ha sido escogida por segunda vez
			{
				*posicionAleatoria = array_posibles_numeros[rand() % 2];	// Cambio ese cero por un numero aleatorio entre los candidatos (2 o 4)
				contadorSemillas++;		// Sumo uno al contador de semillas
			}
		}

	}
	else
	{
		int array_posibles_numeros[] = { 2,4,8 };

		while ((contadorSemillas < 15) && (contadorSemillas < dimMatriz.numFilas * dimMatriz.numColumnas))	// Mientras no se hayan lanzado todas las semillas
		{
			posicionAleatoria = h_matriz + (rand() % dimMatriz.numColumnas) * dimMatriz.numColumnas + (rand() % dimMatriz.numFilas);	// Calculo una posicion aleatoria donde poner una de las semillas
			if (*posicionAleatoria == 0)	// Si es 0 inicialmente, es decir, no ha sido escogida por segunda vez
			{
				*posicionAleatoria = array_posibles_numeros[rand() % 3]; // Cambio ese cero por un numero aleatorio entre los candidatos (2, 4 u 8)
				contadorSemillas++;	// Sumo uno al contador de semillas
			}
		}
	}
}

__host__ void rellenarMatrizconcero(int *h_matriz)
/*
Dada una matriz, la rellena con 0s.
*/
{

	for (int i = 0; i < dimMatriz.numColumnas; ++i) {
		for (int j = 0; j < dimMatriz.numFilas; ++j) {
			*(h_matriz + i * dimMatriz.numColumnas + j) = 0;
		}
	}
}

__host__ void copiarMatriz(int *h_matrizCopia, int *h_matrizPega)
/*
Copia matriz de copia en matriz de pega.
*/
{
	// Punteros a matrices en DEVICE:
	int *d_matrizCopia;
	int *d_matrizPega;

	dimensionesMatriz* d_dimMatriz;

	// Reservo memoria en DEVICE:
	cudaMalloc((void **)&d_matrizCopia, bytesMatriz);
	cudaMalloc((void **)&d_matrizPega, bytesMatriz);
	cudaMalloc((void **)&d_dimMatriz, sizeof(dimensionesMatriz));

	// Muevo matrices de HOST a DEVICE:
	cudaMemcpy(d_matrizCopia, h_matrizCopia, bytesMatriz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrizPega, h_matrizPega, bytesMatriz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dimMatriz, &dimMatriz, sizeof(dimensionesMatriz), cudaMemcpyHostToDevice);

	// Primero, copio salida a entrada. 
	kernelCopiarMatriz << < dimGrid, dimBlock >> > (d_matrizCopia, d_matrizPega, d_dimMatriz);
	cudaDeviceSynchronize();

	// Despu�s, pongo a 0 la matriz de copia. 
	kernelSetMatrizCeros << < dimGrid, dimBlock >> > (d_matrizCopia, d_dimMatriz);
	cudaDeviceSynchronize();

	// Devolvemos resultado de DEVICE a HOST:
	cudaMemcpy(h_matrizPega, d_matrizPega, bytesMatriz, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_matrizCopia, d_matrizCopia, bytesMatriz, cudaMemcpyDeviceToHost);

	// Libero memoria de DEVICE:
	cudaFree(d_matrizPega);
	cudaFree(d_matrizCopia);
	cudaFree(d_dimMatriz);
}

__host__ bool desplazarMatriz(int *h_matrizEntrada, int *h_matrizSalida, int* d_matrizEntrada, int* d_matrizSalida, int* h_puntuacion, int despVertical, int despHorizontal)
{

	int* d_despVertical = 0;
	int* d_despHorizontal = 0;
	int* d_puntuacion = 0;

	dimensionesMatriz* d_dimMatriz;

	// Reservo memoria en DEVICE:
	cudaMalloc((void **)&d_despVertical, sizeof(int));
	cudaMalloc((void **)&d_despHorizontal, sizeof(int));
	cudaMalloc((void **)&d_puntuacion, sizeof(int));
	cudaMalloc((void **)&d_dimMatriz, sizeof(dimensionesMatriz));

	// Muevo matrices de HOST a DEVICE:
	cudaMemcpy(d_matrizEntrada, h_matrizEntrada, bytesMatriz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrizSalida, h_matrizSalida, bytesMatriz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_puntuacion, h_puntuacion, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dimMatriz, &dimMatriz, sizeof(dimensionesMatriz), cudaMemcpyHostToDevice);

	cudaMemcpy(d_despVertical, &despVertical, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_despHorizontal, &despHorizontal, sizeof(int), cudaMemcpyHostToDevice);

	// Realizo la suma:
	kernelSuma << < dimGrid, dimBlock >> > (d_matrizEntrada, d_matrizSalida, d_puntuacion, d_despVertical, d_despHorizontal, d_dimMatriz);

	// Espero a que termine de operar:
	cudaDeviceSynchronize();

	cudaMemcpy(h_puntuacion, d_puntuacion, sizeof(int), cudaMemcpyDeviceToHost);

	// Variable que dice si las matrices son iguales o no.
	bool h_iguales = true;
	bool *d_iguales;
	cudaMalloc((void **)&d_iguales, sizeof(bool));

	// Mientras la matriz de entrada sea distinta de salida,
	// significa que puedo seguir desplazando.
	// Cuando sean iguales, detengo el bucle. 
	do
	{
		// Primero, copio salida a entrada. 
		kernelCopiarMatriz << < dimGrid, dimBlock >> > (d_matrizSalida, d_matrizEntrada, d_dimMatriz);
		cudaDeviceSynchronize();

		// Segundo, seteo salida a 0.
		kernelSetMatrizCeros << < dimGrid, dimBlock >> > (d_matrizSalida, d_dimMatriz);
		cudaDeviceSynchronize();

		// Desplazo
		kernelDesplazar << < dimGrid, dimBlock >> > (d_matrizEntrada, d_matrizSalida, d_despVertical, d_despHorizontal, d_dimMatriz);
		cudaDeviceSynchronize();

		// Compruebo si tengo que seguir desplazando.
		// Doy por hecho que son iguales. Si no lo son, desplazare.
		h_iguales = true;

		// Muevo a device. 
		cudaMemcpy(d_iguales, &h_iguales, sizeof(bool), cudaMemcpyHostToDevice);

		// Veo si son iguales.
		kernelComprobarIguales << < dimGrid, dimBlock  >> > (d_matrizSalida, d_matrizEntrada, d_iguales, d_dimMatriz);
		cudaDeviceSynchronize();

		// Limpio memoria tras trastear con d_iguales. 
		cudaMemcpy(&h_iguales, d_iguales, sizeof(bool), cudaMemcpyDeviceToHost);

	} while (!h_iguales);
	cudaFree(d_iguales);

	// Compruebo si la matriz está llena y si se puede mover en cualq. dirección
	bool h_movimientosPosibles = true;

	// Devolvemos resultado de DEVICE a HOST:
	cudaMemcpy(h_matrizSalida, d_matrizSalida, bytesMatriz, cudaMemcpyDeviceToHost);

	// Si esta llena compruebo si hay movimientos posibles
	if (estaLlena(d_matrizSalida))
		h_movimientosPosibles = movimientosPosibles(d_matrizSalida);
	// Si no, añado una nueva semilla a la matriz resultante en host
	else {
		nuevaSemilla(h_matrizSalida, 1);	// Añadimos la nueva semilla

		// Comprobamos si con la nueva semilla anadida, hemos perdido
		cudaMemcpy(d_matrizSalida, h_matrizSalida, bytesMatriz, cudaMemcpyHostToDevice);
		if (estaLlena(d_matrizSalida))
			h_movimientosPosibles = movimientosPosibles(d_matrizSalida);
	}

	// Libero memoria de DEVICE:
	cudaFree(d_despVertical);
	cudaFree(d_despHorizontal);
	cudaFree(d_dimMatriz);

	return h_movimientosPosibles;
}

__host__ void setElementoHost(int *h_matriz, int fila, int columna, int elemento)
/*
Dada una matriz, escribe el elemento en [fila][columna]
*/
{
	h_matriz[fila * dimMatriz.numColumnas + columna] = elemento;
}

__host__ void nuevaSemilla(int *h_matriz, int numSemillas)
/*
Crea numSemillas nuevas semillas en la matriz almacenada en device
*/
{
	int *posicionAleatoria;
	bool semillaGenerada = false;

	if (modoDiablo)
	{
		int array_posibles_numeros[] = { 2,4 };

		while ((!semillaGenerada) && (numSemillas != 0))	// Mientras no se haya encontrado una posicion con 0 y no se hallan lanzado todas las semillas
		{
			posicionAleatoria = h_matriz + (rand() % dimMatriz.numColumnas) * dimMatriz.numColumnas + (rand() % dimMatriz.numFilas);	// Calculo una posicion aleatoria donde poner una de las semillas
			if (*posicionAleatoria == 0) // Si es 0 inicialmente, es decir, no ha sido escogida por segunda vez
			{
				*posicionAleatoria = array_posibles_numeros[rand() % 2];	// Cambio ese cero por un numero aleatorio entre los candidatos (2 o 4)
				semillaGenerada = true;
				numSemillas--;
			}
		}

	}
	else
	{
		int array_posibles_numeros[] = { 2,4,8 };

		while ((!semillaGenerada) && (numSemillas != 0))	// Mientras no se haya encontrado una posicion con 0 y no se hayan lanzado todas las semillas
		{
			posicionAleatoria = h_matriz + (rand() % dimMatriz.numColumnas) * dimMatriz.numColumnas + (rand() % dimMatriz.numFilas);	// Calculo una posicion aleatoria donde poner una de las semillas
			if (*posicionAleatoria == 0)	// Si es 0 inicialmente, es decir, no ha sido escogida por segunda vez
			{
				*posicionAleatoria = array_posibles_numeros[rand() % 3]; // Cambio ese cero por un numero aleatorio entre los candidatos (2, 4 u 8)
				semillaGenerada = true;
				numSemillas--;
			}
		}
	}
}

// ------- COMPROBACIONES EN HOST ------- //

__host__ bool estaLlena(int* d_matriz)
{
	// Compruebo si la matriz esta llena 
	bool h_estaLlena = true;
	bool *d_estaLlena;
	dimensionesMatriz* d_dimMatriz;

	cudaMalloc((void **)&d_estaLlena, sizeof(bool));
	cudaMalloc((void **)&d_dimMatriz, sizeof(dimensionesMatriz));

	cudaMemcpy(d_estaLlena, &h_estaLlena, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dimMatriz, &dimMatriz, sizeof(dimensionesMatriz), cudaMemcpyHostToDevice);

	// Veo si está llena.
	kernelComprobarLlena << < dimGrid, dimBlock >> > (d_matriz, d_estaLlena, d_dimMatriz);
	cudaDeviceSynchronize();

	cudaMemcpy(&h_estaLlena, d_estaLlena, sizeof(bool), cudaMemcpyDeviceToHost);
	// Limpio memoria tras trastear con d_estaLlena. 
	cudaFree(d_estaLlena);
	cudaFree(d_dimMatriz);

	return h_estaLlena;
}

__host__ bool finJuego(int* d_matriz)
{
	// Compruebo si la matriz contiene algún 16384
	bool h_haGanado = false;
	bool *d_haGanado;
	dimensionesMatriz* d_dimMatriz;

	cudaMalloc((void **)&d_haGanado, sizeof(bool));
	cudaMalloc((void **)&d_dimMatriz, sizeof(dimensionesMatriz));

	cudaMemcpy(d_haGanado, &h_haGanado, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dimMatriz, &dimMatriz, sizeof(dimensionesMatriz), cudaMemcpyHostToDevice);

	// Veo si está llena.
	kernelComprobarSiHaGanado << < dimGrid, dimBlock >> > (d_matriz, d_haGanado, d_dimMatriz);
	cudaDeviceSynchronize();

	cudaMemcpy(&h_haGanado, d_haGanado, sizeof(bool), cudaMemcpyDeviceToHost);
	// Limpio memoria tras trastear con d_estaLlena. 
	cudaFree(d_haGanado);
	cudaFree(d_dimMatriz);

	return h_haGanado;
}

__host__ bool movimientosPosibles(int* d_matriz)
/*
Llama al kernel de comprobacion de movimientos posibles
*/
{
	bool h_movimientosPosibles = false;

	dimensionesMatriz* d_dimMatriz;

	bool *d_movimientosPosibles;
	cudaMalloc((void **)&d_movimientosPosibles, sizeof(bool));
	cudaMalloc((void **)&d_dimMatriz, sizeof(dimensionesMatriz));

	cudaMemcpy(d_movimientosPosibles, &h_movimientosPosibles, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dimMatriz, &dimMatriz, sizeof(dimensionesMatriz), cudaMemcpyHostToDevice);


	// Compruebo si hay movimientos que se puedan hacer
	kernelComprobarMovimientosPosibles << < dimGrid, dimBlock >> > (d_matriz, d_movimientosPosibles, d_dimMatriz);
	cudaDeviceSynchronize();

	// Paso el booleano a memoria del host y libero la memoria de device
	cudaMemcpy(&h_movimientosPosibles, d_movimientosPosibles, sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(d_dimMatriz);
	cudaFree(d_movimientosPosibles);

	return h_movimientosPosibles;
}


// ----- GUARDADO Y LECTURA ----- // 

__host__ void escribirMatriz(int* h_matriz, string nombreJugador, int* puntuacion, int* movimientos)
{
	FILE *archivo;

	// Preparo nombre:
	nombreJugador += ".txt";
	char * nombreArchivo = new char[nombreJugador.length() + 1];
	strcpy(nombreArchivo, nombreJugador.c_str());

	// Abro archivo:
	archivo = fopen(nombreArchivo, "w");
	if (archivo == NULL)
	{
		cout << "Error escribiendo partida. " << endl;
	}
	else
	{
		fprintf(archivo, "%d\n", dimMatriz.numFilas);
		fprintf(archivo, "%d\n", dimMatriz.numColumnas);
		fprintf(archivo, "%d\n", vidas);
		fprintf(archivo, "%d\n", *movimientos);
		fprintf(archivo, "%d\n", *puntuacion);

		for (int i = 0; i < dimMatriz.numColumnas; ++i) {
			for (int j = 0; j < dimMatriz.numFilas; ++j) {
				fprintf(archivo, "%d ", *(h_matriz + i * dimMatriz.numColumnas + j));
			}
			fprintf(archivo, "\n");
		}

	}

	fclose(archivo);
}

__host__ bool leerMatriz(int* h_matriz, string nombreJugador, int* puntuacion, int* movimientos)
{
	// Cargo el archivo
	ifstream in(nombreJugador + ".txt");

	bool lecturaCorrecta = true;

	// Si error
	if (!in)
	{
		cout << "Erro abriendo el archivo. La partida no existe, se iniciara una partida nueva." << endl;
		lecturaCorrecta = false;
	}
	// Si no, escribo matriz
	else
	{
		int a_filas, a_columnas;
		in >> a_filas;
		in >> a_columnas;
		in >> vidas;
		if (a_filas != dimMatriz.numFilas || a_columnas != dimMatriz.numColumnas)
		{
			cout << "La partida cargada no es congruente con el numero de filas/columnas pasada como parametro." << endl;
			cout << "Se iniciara una partida nueva." << endl;
			lecturaCorrecta = false;
		}
		else
		{
			// Cargo movimientos y puntuacion
			in >> *movimientos;
			in >> *puntuacion;
			for (int fila = 0; fila < dimMatriz.numFilas; fila++)
			{
				for (int columna = 0; columna < dimMatriz.numColumnas; columna++)
				{
					// Parseo el numero
					int num;
					in >> num;

					// Lo escribo en la posicion
					setElementoHost(h_matriz, fila, columna, num);
				}
			}
		}
	}

	// Cierro archivo
	in.close();

	return lecturaCorrecta;
}



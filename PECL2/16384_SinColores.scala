object Pec
{
    val random = new scala.util.Random()

    // Dificultad del juego (de 1 a 4)
    val dificultad: Int = iniciarJuego()

    // Dimensión de la matriz.
    val dimension: Int = dimensionPorDificultad()

    // Nº de semillas con las que se inicia el tablero.
    val numSemillasIniciales: Int = numeroSemillasPorDificultad()

    // Semillas posibles.
    val semillasPosibles: List[Int] = semillasPosiblesPorDificultad()

    // Objetivo del juego.
    val objetivo = 16384

    /* 0 - Funciones de tratamiento de listas */

    /**
      * Dado el nº de casillas que tendrá el tablero, devuelve
      * una lista de ese tamaño con todas las seeds a 0.
      */
    def listaDeCeros(nceros: Int): List[Int] = nceros match
    {
        case 0 => Nil
        case ncasillas => listaDeCeros(ncasillas - 1) ::: List(0)
    }


    /**
      * Devuelve True si la lista está vacía.
      */
    def isEmpeti(l: List[Int]): Boolean = l match
    {
        case Nil => true
        case x :: y => false
    }

    /**
      * Dada una lista, devuelve la misma lista quitando los 0s.
      */
    def strip(l: List[Int]): List[Int] = l match
    {
        case Nil => List()
        case x :: y if (x == 0) => strip(y)
        case x :: y if (x != 0) => List(x) ::: strip(y)
    }


    /**
      * Dada una lista, la devuelve invertida.
      */
    def revertir(l: List[Int]): List[Int] = l match
    {
        case Nil => List()
        case x :: y => revertir(y) ::: List(x)
        case x => x
    }


    /**
      * Dada una lista, devuelve una sublista que comprende las posiciones
      * lista[desde:hasta], ambas incluidas.
      */
    def recortar(l: List[Int], desde: Int, hasta: Int): List[Int] = l match
    {
        case Nil => List()
        // Si me paso, lanzo error.
        case _ if (longitud(l) < hasta) =>
            throw new Error("Te has pasado de la longitud de la lista recortando crack. ")

        case _ if (desde < 0 || desde > hasta) => throw new Error("No se admiten desde's negativos.")
        case x :: y if (desde > 0) => recortar(y, desde - 1, hasta - 1)
        case x :: y if (desde == 0 && hasta > 0) => x :: recortar(y, 0, hasta - 1)
        case x :: y if (desde == 0 && hasta == 0) => List()
    }

    /**
      * Devuelve la longitud (Int) de la lista.
      */
    def longitud(l: List[Int]): Int = l match
    {
        case Nil => 0
        case x :: y => 1 + longitud(y)
    }

    /**
      * Elimina la primera instancia del elemento en la lista.
      */
    def eliminar(lista: List[Int], elemento: Int): List[Int] = lista match
    {
        case Nil => List()
        case x :: y if (x == elemento) => y
        case x :: y if (x != elemento) => x :: eliminar(y, elemento)
    }

    /**
      * Dadas dos listas, devuelve la disyunción entre estas.
      */
    def disyuncion(l1: List[Int], l2: List[Int]): List[Int] = l1 match
    {
        case Nil => l2
        case x :: y if (contiene(l2, x)) => disyuncion(y, eliminar(l2, x))
        case x :: y if (!contiene(l2, x)) => x :: disyuncion(y, l2)
    }

    /**
      * Devuelve el elemento en la posición dada.
      */
    def getElemento(l: List[Int], pos: Int): Int = l match
    {
        case Nil => throw new Error("Te has pasado de indice en la lectura crack.")
        case x :: y if (pos == 0) => x
        case x :: y if (pos != 0) => getElemento(y, pos - 1)
    }


    /**
      * Escribe el valor en la posición dada.
      */
    def setElemento(l: List[Int], pos: Int, valor: Int): List[Int] = l match
    {
        case Nil => throw new Error("Te has pasado de indice en la escritura crack.")
        case x :: y if (pos == 0) => valor :: y
        case x :: y if (pos != 0) => x :: setElemento(y, pos - 1, valor)
    }

    /**
      * Devuelve true si el elemento se encuentra en la lista.
      */
    def contiene(lista: List[Int], elemento: Int): Boolean = lista match
    {
        case Nil => false
        case `elemento` :: y => true
        case _ :: y => contiene(y, elemento)
    }

    /**
      * Devuelve el nº de apariciones del elemento en la lista dada.
      */
    def cuantosContiene(l: List[Int], elemento: Int): Int = l match
    {
        case Nil => 0
        case `elemento` :: y => 1 + cuantosContiene(y, elemento)
        case _ :: y => cuantosContiene(y, elemento)
    }

    /**
      * Devuelve cuántos elementos son distintos de 0 en una lista.
      */
    def cuantosDistintosDeCero(l: List[Int]): Int = l match
    {
        case Nil => 0
        case 0 :: y => cuantosDistintosDeCero(y)
        case _ :: y => 1 + cuantosDistintosDeCero(y)
    }

    /**
      * Dadas dos listas, devuelve un booleano que indica si
      * son iguales.
      */
    def sonIguales(l1: List[Int], l2: List[Int]): Boolean = l1 match
    {
        case _ if(longitud(l1) != longitud(l2)) => throw new Error("Listas incorrectas. ")
        case Nil => true
        case x :: _ if (x == l2.head) => sonIguales(l1.tail, l2.tail)
        case x :: _ if (x != l2.head) => false
    }

    def maximo(l: List[Int]): Int = l match
    {
        case Nil => 0
        case _ => Math.max(l.head, maximo(l.tail))
    }


    /* 1 - Funciones de inicio de juego que no deberíamos tener que tocar */

    /**
      * Da la bienvenida a pide dificultad al jugador.
      */
    def iniciarJuego(): Int =
    {
        println("Hola amigo bienvenido al 16384 vaya ganitas de jugar. ")

        pedirDificultad()
    }


    /**
      * Pide dificultad por input. Si se meten valores incorrectos, se llama recursivamente.
      */
    def pedirDificultad(): Int =
    {
        try
        {
            println("¿Que nivel de dificultad quieres?(Entre 1 y 4): ")
            scala.io.StdIn.readInt() match
            {
                case x if x > 4 || x < 1 => println("Numero no valido")
                    pedirDificultad()
                case x => x
            }
        }
        catch
        {
            case _: java.lang.NumberFormatException => println("El valor introducido no es un numero")
                pedirDificultad()
        }
    }


    /**
      * Determina las dimensiones del tablero en base a la dificultad elegida
      */
    def dimensionPorDificultad(): Int = dificultad match
    {
        case 1 => 4
        case 2 => 9
        case 3 => 14
        case 4 => 17
    }

    /**
      * Determina el numero de semillas que se tienen que generar despues de cada jugada
      * en base a la dificultad elegida
      */
    def numeroSemillasPorDificultad(): Int = dificultad match
    {
        case 1 => 2
        case 2 => 4
        case _ => 6
    }


    /**
      * Determina las semillas que se deben generar segun el nivel de
      * dificultad elegido
      */
    def semillasPosiblesPorDificultad(): List[Int] = dificultad match
    {
        case 1 => List(2)
        case 2 => List(2, 4)
        case _ => List(2, 4, 8)
    }

    /**
      * Dado un tablero vacío, lo rellena como un pavo en navidad con el número
      * de seeds que se diga y las elige de la lista de seeds dada.
      */
    def rellenarTableroVacio(tablero: List[Int], cuantasSeeds: Int, posiblesSeeds: List[Int]): List[Int] = cuantasSeeds match
    {

        case 0 => tablero
        case _ =>
            val posicion = random.nextInt(longitud(tablero))
            val noHayCero = getElemento(tablero, posicion) != 0

            // Si la posición aleatoria contiene algo distinto de 0, llamo de nuevo a iniciarTablero.
            if (noHayCero)
            {
                rellenarTableroVacio(tablero, cuantasSeeds, posiblesSeeds)
            }
            else
            {
                val seed = getElemento(posiblesSeeds, random.nextInt(longitud(posiblesSeeds)))
                rellenarTableroVacio(setElemento(tablero, posicion, seed), cuantasSeeds - 1, posiblesSeeds)
            }

    }

    /**
      * Devuelve un tablero en función de la dificultad establecida.
      */
    def crearTablero(dificultad: Int): List[Int] = dificultad match
    {
        case 1 => rellenarTableroVacio(listaDeCeros(dimension * dimension), numSemillasIniciales, semillasPosibles)
        case 2 => rellenarTableroVacio(listaDeCeros(dimension * dimension), numSemillasIniciales, semillasPosibles)
        case 3 => rellenarTableroVacio(listaDeCeros(dimension * dimension), numSemillasIniciales, semillasPosibles)
        case 4 => rellenarTableroVacio(listaDeCeros(dimension * dimension), numSemillasIniciales, semillasPosibles)
    }


    /* 2 - Funciones de tratamiento de tablero */

    // Getters:

    def getColumnaAux(tablero: List[Int], columna: Int): List[Int] = columna match
    {
        // Si me paso del tablero
        case x if (x >= (dimension * dimension)) => List()
        // Si no me paso
        case _ =>
            getElemento(tablero, columna) :: getColumnaAux(tablero, columna + dimension)
    }

    /**
      * Dado un tablero y una columna, devuelve dicha columna.
      */
    def getColumna(tablero: List[Int], columna: Int): List[Int] = columna match
    {
        case x if (x > dimension - 1) => throw new Error("No hay tantas columnas crack. Recuerda que hay columna 0.")
        case x if (x < 0) => throw new Error("No hay columnas negativas crack.")
        case _ => getColumnaAux(tablero, columna)
    }

    /**
      * Dado un tablero y un numero de fila, devuelve dicha fila.
      */
    def getFila(tablero: List[Int], fila: Int): List[Int] = fila match
    {
        case x if (x > (dimension - 1)) => throw new Error("No hay tantas filas crack. Recuerda que hay fila 0.")
        case 0 => recortar(tablero, 0, dimension)
        case _ =>
            recortar(tablero, fila * dimension, fila * dimension + dimension)
    }

    // Setters:

    /**
      * Dado un tablero, un numero de columna y una nueva columna a escribir,
      * sustituya la columna indicada por la nueva.
      */
    def setColumna(tablero: List[Int], numColumna: Int, nuevaColumna: List[Int]): List[Int] = numColumna match
    {
        case x if (x > dimension - 1) => throw new Error("No hay tantas columnas crack. Recuerda que hay columna 0.")
        case x if (x < 0) => throw new Error("No hay columnas negativas crack.")
        case x if (longitud(nuevaColumna) != dimension) => throw new Error("Dimensión de columna incorrecta crack.")
        case _ => setColumnaAux(tablero, numColumna, nuevaColumna, 0)
    }

    def setColumnaAux(tablero: List[Int], numColumna: Int, nuevaColumna: List[Int], vanPuestos: Int): List[Int] = vanPuestos match
    {
        // Si es ultima columna, escribo por última vez y detengo la llamada recursiva.
        case x if (x == `dimension` - 1) =>
            setElemento(tablero, numColumna + dimension * vanPuestos, getElemento(nuevaColumna, vanPuestos))
        // Si no es la última columna, escribo el dato y llamo recursivamente para el siguiente.
        case _ =>
            setColumnaAux(
                // Escribo
                setElemento(tablero, numColumna + dimension * vanPuestos, getElemento(nuevaColumna, vanPuestos)),
                // Y llamo al siguiente
                numColumna, nuevaColumna, vanPuestos + 1)
    }

    /**
      * Dado un tablero, una fila indicada y una nueva fila,
      * sustituye esa fila indicada por la nueva pasada.
      */
    def setFila(tablero: List[Int], numFila: Int, nuevaFila: List[Int]): List[Int] = numFila match
    {
        case x if (x > dimension - 1) => throw new Error("No hay tantas filas crack. Recuerda que hay columna 0.")
        case x if (x < 0) => throw new Error("No hay filas negativas crack.")
        case x if (longitud(nuevaFila) != dimension) => throw new Error("Dimensión de fila incorrecta crack.")
        case _ =>
            recortar(tablero, 0, (dimension) * numFila) :::
                nuevaFila :::
                recortar(tablero, dimension + numFila * dimension, dimension * dimension)
    }

    // Comprobadores:

    /**
      * Devuelve true si el elemento de debajo de la posición es el mismo
      * que el de la posición.
      */
    def comprobarAbajo(tablero: List[Int], posicion: Int): Boolean =
    {
        getElemento(tablero, posicion) == getElemento(tablero, posicion + dimension)
    }

    /**
      * Devuelve true si el elemento de la derecha de la posición es el mismo
      * que el de la posición.
      */
    def comprobarDerecha(tablero: List[Int], posicion: Int): Boolean =
    {
        getElemento(tablero, posicion) == getElemento(tablero, posicion + 1)
    }

    /**
      * Comprueba si el tabLero esta lleno explorando recursivamente posicion a posicion
      */
    def estaLleno(tablero: List[Int], posicion: Int): Boolean = posicion match
    {
        case pos if getElemento(tablero, pos) == 0 => false // Si el elemento es 0, el tablero no esta lleno
        case pos if pos == (dimension * dimension) - 1 && getElemento(tablero, pos) != 0 => true // Si esta en la ultima posicion y el valor es distinto de 0, devuelvo true
        case pos if getElemento(tablero, pos) != 0 => estaLleno(tablero, pos + 1) // Si el elemento es distinto de 0, puedo seguir explorando
    }

    /**
      * Comprueba si hay movimientos posibles en el tablero, es decir, si las casillas pueden combinarse con sus adyacentes
      * Para que funcione correctamente, solo se puede llamar a este metodo si se ha comprobado que el tablero este lleno previamente
      */
    def movimientosPosibles(tablero: List[Int], posicion: Int): Boolean = posicion match
    {
        case pos if pos == (dimension * dimension) - 1 => // Si nos encontramos en la ultima posicion es que no hay movimientos posibles
            false

        case pos if (pos / dimension) == (dimension - 1) => // Si nos encontramos en la ultima fila
            if (comprobarDerecha(tablero, pos))
            {
                true
            }
            else
            {
                movimientosPosibles(tablero, pos + 1)
            }

        case pos if (pos % dimension) != (dimension - 1) => // Si no nos encontramos en la ultima columna
            if (comprobarAbajo(tablero, pos) || comprobarDerecha(tablero, pos)) // Si me puedo mover en alguna direccion, hay movimientos posibles
            {
                true
            }
            else
            {
                movimientosPosibles(tablero, pos + 1)
            } // Si no me puedo mover en ninguna direccion, llamo a recursion incrementando la posicion

        case pos if (pos % dimension) == dimension - 1 => // Si nos encontramos en la ultima columna
            if (comprobarAbajo(tablero, pos))
            {
                true
            }
            else
            {
                movimientosPosibles(tablero, pos + 1)
            }
    }

    def seguirJugando(tablero: List[Int], tableroLleno: Boolean): Boolean = tableroLleno match
    {
        case false => true // Si no esta lleno, podemos seguir jugando
        case true if movimientosPosibles(tablero, 0) => true // Si esta lleno pero hay movimientos posibles, podemos seguir jugando
        case _ => false // El otro caso posibles es que este lleno y no haya movimientos posibles, en cuyo caso no podemos seguir jugando
    }

    /* 3 - Operaciones y movimientos */

    // Operaciones a nivel de fila/columna: calculos principales.

    /**
      * Dada una fila stripeada, realiza las operaciones de suma a la derecha.
      */
    def sumarDerechaAux(fila_strip: List[Int]): List[Int] = fila_strip match
    {
        case Nil => List()
        case x :: Nil => List(x)
        case x :: x1 :: y if (x == x1) => x1 * 2 :: sumarDerechaAux(y)
        case x :: x1 :: y if (x != x1) => x :: sumarDerechaAux(x1 :: y)
    }

    /**
      * Función aux. para añadir 0s a la fila tras sumar.
      */
    def desplazarDerechaAux(fila: List[Int]): List[Int] =
    {
        val cuantosElementos = longitud(strip(fila))
        fila ::: listaDeCeros(dimension - cuantosElementos)
    }

    /**
      * Tras sumar, se añaden 0s a la derecha como desplazamiento.
      */
    def operarFila(fila: List[Int]): List[Int] =
    {
        desplazarDerechaAux(sumarDerechaAux(strip(fila)))
    }

    // Puntuación y colisiones:

    def calculaColisiones(tableroAntes: List[Int], tableroDespues: List[Int]): Int =
    {
        longitud(strip(tableroAntes)) - longitud(strip(tableroDespues))
    }

    /**
      * Calcula la puntuación de una fila o columna sin operar.
      */
    def calculaPuntuacion(l: List[Int]): Int = l match
    {
        case Nil => 0
        case x :: Nil => 0
        case x :: y :: z if (x == y) => x * 2 + calculaPuntuacion(z)
        case x :: y :: z if (x != y) => calculaPuntuacion(y :: z)
    }

    def calcularPuntuacionMoverDerecha(tablero: List[Int], fila: Int): Int = fila match
    {
        case x if (x > dimension - 1) => 0
        case _ =>
            calculaPuntuacion(strip(getFila(tablero, fila))) + calcularPuntuacionMoverDerecha(tablero, fila + 1)

    }

    def calcularPuntuacionMoverIzquierda(tablero: List[Int], fila: Int): Int = fila match
    {
        case x if (x > dimension - 1) => 0
        case _ =>
            calculaPuntuacion(strip(revertir(getFila(tablero, fila)))) + calcularPuntuacionMoverIzquierda(tablero, fila + 1)

    }

    def calcularPuntuacionMoverAbajo(tablero: List[Int], columna: Int): Int = columna match
    {
        case x if (x > dimension - 1) => 0
        case _ =>
            calculaPuntuacion(strip(getColumna(tablero, columna))) + calcularPuntuacionMoverAbajo(tablero, columna + 1)

    }

    def calcularPuntuacionMoverArriba(tablero: List[Int], columna: Int): Int = columna match
    {
        case x if (x > dimension - 1) => 0
        case _ =>
            calculaPuntuacion(strip(revertir(getColumna(tablero, columna)))) + calcularPuntuacionMoverArriba(tablero, columna + 1)
    }

    def calcularMejorJugadaAux(listaPuntuaciones: List[Int], indice: Int): Int = listaPuntuaciones match
    {
        case Nil => 0
        case _ =>
            if (maximo(listaPuntuaciones) == 0)
            {
                random.nextInt(4)
            }
            else if (maximo(listaPuntuaciones) == listaPuntuaciones.head)
            {
                indice
            }
            else
            {
                calcularMejorJugadaAux(listaPuntuaciones.tail, indice + 1)
            }
    }

    def calcularMejorJugadaFutura(tablero: List[Int]): Int =
    {
        maximo(
            List
            (
                calcularPuntuacionMoverIzquierda(tablero, 0),
                calcularPuntuacionMoverDerecha(tablero, 0),
                calcularPuntuacionMoverArriba(tablero, 0),
                calcularPuntuacionMoverAbajo(tablero, 0)
            ))
    }

    /** Devuelve la mejor jugada */
    def calcularMejorJugada(tablero: List[Int]): Int =
    {
        calcularMejorJugadaAux(
            List
            (
                if (calcularPuntuacionMoverIzquierda(tablero, 0) == 0)
                {
                    calcularPuntuacionMoverIzquierda(tablero, 0)
                }
                else
                {
                    calcularPuntuacionMoverIzquierda(tablero, 0) + calcularMejorJugadaFutura(moverTableroIzquierda(tablero, 0))
                },

                if (calcularPuntuacionMoverDerecha(tablero, 0) == 0)
                {
                    calcularPuntuacionMoverDerecha(tablero, 0)
                }
                else
                {
                    calcularPuntuacionMoverDerecha(tablero, 0) + calcularMejorJugadaFutura(moverTableroDerecha(tablero, 0))
                },

                if (calcularPuntuacionMoverArriba(tablero, 0) == 0)
                {
                    calcularPuntuacionMoverArriba(tablero, 0)
                }
                else
                {
                    calcularPuntuacionMoverArriba(tablero, 0) + calcularMejorJugadaFutura(moverTableroArriba(tablero, 0))
                },
                if (calcularPuntuacionMoverAbajo(tablero, 0) == 0)
                {
                    calcularPuntuacionMoverAbajo(tablero, 0)
                }
                else
                {
                    calcularPuntuacionMoverAbajo(tablero, 0) + calcularMejorJugadaFutura(moverTableroAbajo(tablero, 0))
                }

            ),
            0)
    }

    // Movimientos de tablero:

    def moverTableroDerecha(tablero: List[Int], fila: Int): List[Int] = fila match
    {
        case x if (x > dimension - 1) => tablero
        case _ =>
            moverTableroDerecha(setFila(tablero, fila, revertir(operarFila(revertir(getFila(tablero, fila))))), fila + 1)
    }

    def moverTableroIzquierda(tablero: List[Int], fila: Int): List[Int] = fila match
    {
        case x if (x > dimension - 1) => tablero
        case _ =>
            moverTableroIzquierda(setFila(tablero, fila, operarFila(getFila(tablero, fila))), fila + 1)
    }

    def moverTableroAbajo(tablero: List[Int], columna: Int): List[Int] = columna match
    {
        case x if (x > dimension - 1) => tablero
        case _ =>
            moverTableroAbajo(setColumna(tablero, columna, revertir(operarFila(revertir(getColumna(tablero, columna))))), columna + 1)
    }

    def moverTableroArriba(tablero: List[Int], columna: Int): List[Int] = columna match
    {
        case x if (x > dimension - 1) => tablero
        case _ =>
            moverTableroArriba(setColumna(tablero, columna, operarFila(getColumna(tablero, columna))), columna + 1)
    }

    /* 4 - Funciones auxiliares de juego */

    /**
      * Dibuja por pantalla el tablero pasado.
      */
    def pintarTableroAux(lista: List[Int], columna: Int): Unit =
    {
        if (lista != Nil)
        {
            print("[\t"+lista.head)
            if (lista.head % 1000 == lista.head)
                print("\t")

            print("]")

            if (columna + 1 == dimension)
            {
                print("\n")
            }
            pintarTableroAux(lista.tail, (columna + 1) % dimension)
        }
    }

    def pintarTablero(tablero: List[Int]): Unit =
    {
        pintarTableroAux(tablero, 0)
    }

    def addSemillas(tablero: List[Int]): List[Int] =
    {
        dificultad match
        {
            case 1 => addSemillasAux(tablero, 1)
            case 2 => addSemillasAux(tablero, 3)
            case 3 => addSemillasAux(tablero, 5)
            case 4 => addSemillasAux(tablero, 6)
        }
    }

    /**
      * Dado un tablero, retorna un tablero con el numero de seeds dado añadidas, eligiendo
      * de posiblesSeeds.
      */
    def addSemillasAux(tablero: List[Int], cuantasSeeds: Int): List[Int] = cuantasSeeds match
    {
        case 0 => tablero
        case _ =>
            if (!estaLleno(tablero, 0))
            {
                // Obtengo posición actual y valor en esa posición, y genero seed
                val posicion = random.nextInt(longitud(tablero))
                val seed = getElemento(semillasPosibles, random.nextInt(longitud(semillasPosibles)))
                val hayCero = getElemento(tablero, posicion) == 0


                if (!hayCero)
                {
                    addSemillasAux(tablero, cuantasSeeds)
                }
                else
                {
                    addSemillasAux(setElemento(tablero, posicion, seed), cuantasSeeds - 1)
                }
            }
            else
            {
                tablero
            }
    }

    // Funciones de verificación de I/O
    /**
      * Devuelve a (automático) o m(manual), o pide que se vuelva a
      * introducir caracter en caso de entrada incorrecta.
      */
    def modoJuego(): String =
    {
        println("¿Modo manual o automatico?(m/a)")
        scala.io.StdIn.readLine().toLowerCase match
        {
            case "a" => "a"
            case "m" => "m"
            case _ => println("La tecla introducida no es valida. ")
                modoJuego()
        }
    }

    /**
      * Devuelve s(si) o n(no), o pide que se vuelva a
      * introducir caracter en caso de entrada incorrecta.
      */
    def siONo(): String =
    {
        println("¿Quieres jugar otra vez? (s/n)")
        scala.io.StdIn.readLine().toLowerCase match
        {
            case "s" => "s"
            case "n" => "n"
            case _ => println("La tecla introducida no es valida. ")
                siONo()
        }
    }

    /* 4 - Funciones de juego principal */

    def bucleJuegoManual(tablero: List[Int], numMovimientos: Int, puntuacion: Int, colisiones: Int): Boolean =
    {
        println("Movimientos:\t " + numMovimientos + "\tPuntuacion:\t" + puntuacion + "\tColisiones:\t" + colisiones)
        pintarTablero(tablero)

        // Si ganas, tiras true
        if (contiene(tablero, objetivo))
        {
            println("¡Felicidades! Has ganado con: " + puntuacion + " puntos.")
            true
        }
        // Si no, si puedes seguir jugando
        else if (seguirJugando(tablero, estaLleno(tablero, 0)))
        {
            scala.io.StdIn.readLine().toLowerCase match
            {
                case "a" => println("Izquierda. ")
                    if (sonIguales(moverTableroIzquierda(tablero, 0), tablero))
                    {
                        println("Movimiento no valido. ")
                        bucleJuegoManual(tablero, numMovimientos, puntuacion, colisiones)
                    }
                    else
                    {
                        bucleJuegoManual(addSemillas(moverTableroIzquierda(tablero, 0)), numMovimientos + 1, puntuacion + calcularPuntuacionMoverIzquierda(tablero, 0),
                            colisiones + calculaColisiones(tablero, moverTableroIzquierda(tablero, 0)))
                    }

                case "d" => println("Derecha. ")
                    if (sonIguales(moverTableroDerecha(tablero, 0), tablero))
                    {
                        println("Movimiento no valido. ")
                        bucleJuegoManual(tablero, numMovimientos, puntuacion, colisiones)
                    }
                    else
                    {
                        bucleJuegoManual(addSemillas(moverTableroDerecha(tablero, 0)), numMovimientos + 1, puntuacion + calcularPuntuacionMoverDerecha(tablero, 0),
                            colisiones + calculaColisiones(tablero, moverTableroDerecha(tablero, 0)))

                    }

                case "w" => println("Arriba. ")
                    if (sonIguales(moverTableroArriba(tablero, 0), tablero))
                    {
                        println("Movimiento no valido. ")
                        bucleJuegoManual(tablero, numMovimientos, puntuacion, colisiones)
                    }
                    else
                    {
                        bucleJuegoManual(
                            // Nuevo tablero
                            addSemillas(moverTableroArriba(tablero, 0)),
                            // Movimientos
                            numMovimientos + 1,
                            // Puntuación
                            puntuacion + calcularPuntuacionMoverArriba(tablero, 0),
                            // Colisiones
                            colisiones + calculaColisiones(tablero, moverTableroArriba(tablero, 0)))
                    }

                case "s" => println("Abajo")
                    if (sonIguales(moverTableroAbajo(tablero, 0), tablero))
                    {
                        println("Movimiento no valido. ")
                        bucleJuegoManual(tablero, numMovimientos, puntuacion, colisiones)
                    }
                    else
                    {
                        bucleJuegoManual(addSemillas(moverTableroAbajo(tablero, 0)), numMovimientos + 1, puntuacion + calcularPuntuacionMoverAbajo(tablero, 0),
                            colisiones + calculaColisiones(tablero, moverTableroAbajo(tablero, 0)))
                    }

                case "x" => println("Has conseguido: " + puntuacion + " puntos")
                    println("¿Te rindes?\n¡Hasta luego!")
                    sys.exit(0)
                case _ => println("La tecla pulsada no es valida")
                    bucleJuegoManual(tablero, numMovimientos, puntuacion, colisiones)
            }
        }
        else
        {
            println("No hay mas movimientos posibles, has perdido. Tu puntuación es de " + puntuacion + ". Gracias por jugar!")
            false
        }
    }

    def bucleJuegoAutomatico(tablero: List[Int], numMovimientos: Int, puntuacion: Int, colisiones: Int): Boolean =
    {
        println("Movimientos:\t " + numMovimientos + "\tPuntuacion:\t" + puntuacion + "\tColisiones:\t" + colisiones)
        pintarTablero(tablero)
        println("Pulsa Enter para seguir / x para salir")

        if (contiene(tablero, objetivo))
        {
            println("La máquina ha ganado con: " + puntuacion + " puntos.")
            true
        }
        else if (seguirJugando(tablero, estaLleno(tablero, 0)))
        {
            scala.io.StdIn.readLine().toLowerCase match
            {
                case "x" => println("La máquina tenía " + puntuacion + " puntos.")
                    println("¡Hasta luego!")
                    sys.exit(0)
                case _ =>
                    calcularMejorJugada(tablero) match
                    {
                        case 0 => println("Izquierda")
                            bucleJuegoAutomatico(addSemillas(moverTableroIzquierda(tablero, 0)), numMovimientos + 1, puntuacion + calcularPuntuacionMoverIzquierda(tablero, 0),
                                colisiones + calculaColisiones(tablero, moverTableroIzquierda(tablero, 0)))

                        case 1 => println("Derecha")
                            bucleJuegoAutomatico(addSemillas(moverTableroDerecha(tablero, 0)), numMovimientos + 1, puntuacion + calcularPuntuacionMoverDerecha(tablero, 0),
                                colisiones + calculaColisiones(tablero, moverTableroDerecha(tablero, 0)))

                        case 2 => println("Arriba")
                            bucleJuegoAutomatico(addSemillas(moverTableroArriba(tablero, 0)), numMovimientos + 1, puntuacion + calcularPuntuacionMoverArriba(tablero, 0),
                                colisiones + calculaColisiones(tablero, moverTableroArriba(tablero, 0)))

                        case 3 => println("Abajo")
                            bucleJuegoAutomatico(addSemillas(moverTableroAbajo(tablero, 0)), numMovimientos + 1, puntuacion + calcularPuntuacionMoverAbajo(tablero, 0),
                                colisiones + calculaColisiones(tablero, moverTableroAbajo(tablero, 0)))
                    }
            }
        }
        else
        {
            println("No hay más movimientos posibles, la maquina ha perdido con " + puntuacion + " puntos. ")
            false
        }
    }

    def jugandoConVidasManual(ganado: Boolean, vidas: Int, vecesGanado: Int): Unit = ganado match
    {
        case true if (vecesGanado < 3) =>
            println("Has ganado " + (vecesGanado + 1) + " veces seguidas. ")
            siONo() match
            {
                case "s" =>
                    jugandoConVidasManual(bucleJuegoManual(crearTablero(dificultad), 0, 0, 0),
                        vidas + 1, vecesGanado + 1)
                case "n" =>
                    println("Gracias por jugar. Un saludo amigo. ")
            }
        case true if (vecesGanado >= 2) =>
            println("Has ganado 3 veces. Eres el tricampeón. Gracias por jugar. ")
        case false if (vidas > 1) =>
            println("Vaya, eres un perdedor. Aún te quedan vidas, ¿quieres jugar otra vez?")
            siONo() match
            {
                case "s" =>
                    jugandoConVidasManual(bucleJuegoManual(crearTablero(dificultad), 0, 0, 0),
                        vidas - 1, vecesGanado)
                case "n" =>
                    println("Gracias por jugar. Un saludo amigo. ")
            }
        case false if (vidas <= 1) =>
            println("Te has quedado sin vidas. Eres un desastre. Adios. ")

    }

    def jugandoConVidasAutomatico(ganado: Boolean, vidas: Int, vecesGanado: Int): Unit = ganado match
    {
        case true if (vecesGanado < 3) =>
            println("La máquina ha ganado y tiene una vida extra. Vidas: " + (vidas + 1))
            println("Has ganado " + (vecesGanado + 1) + " veces seguidas. ")
            siONo() match
            {
                case "s" =>
                    jugandoConVidasAutomatico(bucleJuegoAutomatico(crearTablero(dificultad), 0, 0, 0),
                        vidas + 1, vecesGanado + 1)
                case "n" =>
                    println("Gracias por jugar. Un saludo amigo. ")
            }
        case true if (vecesGanado >= 2) =>
            println("Has ganado 3 veces. Eres el tricampeón. Gracias por jugar. ")
        case false if (vidas > 1) =>
            println("Vaya, la máquina ha perdido. Aún quedan " + (vidas - 1) + " vidas, ¿quieres jugar otra vez?")
            siONo() match
            {
                case "s" =>
                    jugandoConVidasAutomatico(bucleJuegoAutomatico(crearTablero(dificultad), 0, 0, 0),
                        vidas - 1, vecesGanado)
                case "n" =>
                    println("Gracias por jugar maquinote. Un saludo a turing. ")
            }
        case false if (vidas <= 1) =>
            println("La máquina se ha quedado sin vidas. Es un desastre. Adios. ")

    }

    def main(args: Array[String]): Unit =
    {
        modoJuego() match
        {
            case "m" => jugandoConVidasManual(bucleJuegoManual(crearTablero(dificultad), 0, 0, 0), 3, 0)

            case "a" => jugandoConVidasAutomatico(bucleJuegoAutomatico(crearTablero(dificultad), 0, 0, 0), 3, 0)
        }
    }


}

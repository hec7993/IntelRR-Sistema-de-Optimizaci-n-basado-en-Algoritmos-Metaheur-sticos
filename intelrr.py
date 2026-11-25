"""
IntelRR - Optimizacion de asignacion de recursos en la nube con algoritmos hibridos
===================================================================================

QUe HACE ESTE PROGRAMA
  Hay muchas tareas y varias maquinas virtuales (computadoras en la nube)
  Hay que decidir que tarea se ejecuta en que maquina para que:
     1) todo termine lo mas rapido posible        (tiempo total de ejecucion (makespan))
     2) se gaste la menor energia posible
     3) ninguna maquina se pase de su memoria disponible (si se pasa, hay un castigo)

ALGORITMOS QUE SE COMPARAN
  - Round-Robin y Voraz : metodos sencillos de referencia (no son metaheuristicas)
  - Algoritmo genetico  : imita la evolucion
  - PSO                 : imita una bandada de aves buscando comida
  - Hibrido GA-PSO      : algoritmo genetico con ideas tomadas del PSO

CoMO SE USA
  python intelrr.py            -> experimento completo (tarda 1 o 2 minutos)
  python intelrr.py --rapido   -> version corta para probar que todo funciona

ESTRUCTURA DEL ARCHIVO
  Parte 1: crear el problema y calcular que tan buena es una solucion
  Parte 2: metodos de referencia (Round-Robin y Voraz)
  Parte 3: los tres algoritmos metaheuristicos
  Parte 4: funciones para repetir los experimentos
  Parte 5: graficas
  Parte 6: experimento completo y estadistica
  Parte 7: inicio del programa
"""

import csv
import json
import os
import sys
import time

import numpy
import matplotlib
matplotlib.use("Agg")                    # permite guardar graficas sin abrir ventanas
from matplotlib import pyplot
from scipy import stats


# =============================================================================
# EL PROBLEMA Y LA FORMA DE CALIFICAR UNA SOLUCION
# =============================================================================
# Una solucion es una lista con un numero por cada tarea
# Ese numero indica a que maquina virtual se asigna la tarea
# Ejemplo con 5 tareas y 3 maquinas: [0, 2, 2, 1, 0]
#   -> la tarea 0 va a la maquina 0, la tarea 1 a la maquina 2, etc

def crear_problema(numero_de_tareas, numero_de_maquinas, semilla=2026, peso_del_tiempo=0.5, factor_de_penalizacion=10.0, holgura_de_memoria=1.25):
    #Crea un problema de ejemplo con numeros aleatorios (siempre iguales para la misma semilla)
    generador_aleatorio = numpy.random.default_rng(semilla)

    # Cada tarea tiene una longitud (millones de instrucciones) y necesita memoria (MB)
    longitud_de_cada_tarea = generador_aleatorio.uniform(500, 5000, numero_de_tareas)
    memoria_de_cada_tarea = generador_aleatorio.choice([256, 512, 1024, 2048], numero_de_tareas).astype(float)

    # Cada maquina tiene una velocidad (MIPS) y un consumo electrico (watts)
    velocidad_de_cada_maquina = generador_aleatorio.choice([500, 1000, 1500, 2000, 2500, 3000], numero_de_maquinas).astype(float)
    potencia_en_uso = 40 + 0.05 * velocidad_de_cada_maquina * generador_aleatorio.uniform(0.8, 1.4, numero_de_maquinas)
    potencia_en_reposo = 0.4 * potencia_en_uso

    # La memoria total de las maquinas es 25 % mayor que la que piden las tareas, repartida de forma desigual. Asi el limite de memoria si importa, pero es posible cumplirlo.
    peso_para_repartir_memoria = velocidad_de_cada_maquina * generador_aleatorio.uniform(0.7, 1.3, numero_de_maquinas)
    memoria_total_necesaria = memoria_de_cada_tarea.sum()
    capacidad_de_memoria = (holgura_de_memoria * memoria_total_necesaria * peso_para_repartir_memoria / peso_para_repartir_memoria.sum())

    problema = {
        "numero_de_tareas": numero_de_tareas,
        "numero_de_maquinas": numero_de_maquinas,
        "longitud_de_cada_tarea": longitud_de_cada_tarea,
        "memoria_de_cada_tarea": memoria_de_cada_tarea,
        "velocidad_de_cada_maquina": velocidad_de_cada_maquina,
        "potencia_en_uso": potencia_en_uso,
        "potencia_en_reposo": potencia_en_reposo,
        "capacidad_de_memoria": capacidad_de_memoria,
        "peso_del_tiempo": peso_del_tiempo,
        "peso_de_la_energia": 1.0 - peso_del_tiempo,
        "factor_de_penalizacion": factor_de_penalizacion,
        "tiempo_de_referencia": 1.0,
        "energia_de_referencia": 1.0,
    }

    # Para que tiempo y energia se puedan sumar, se dividen entre valores de referencia
    # (los que da el metodo Round-Robin)
    solucion_round_robin = crear_solucion_round_robin(problema)
    poblacion_de_una_solucion = numpy.array([solucion_round_robin])
    tiempo_ocupado, memoria_usada = calcular_tiempo_y_memoria_por_maquina(problema, poblacion_de_una_solucion)
    tiempo_total, energia, exceso_de_memoria = calcular_metricas(problema, tiempo_ocupado, memoria_usada)
    problema["tiempo_de_referencia"] = float(tiempo_total[0])
    problema["energia_de_referencia"] = float(energia[0])
    return problema

def calcular_tiempo_y_memoria_por_maquina(problema, poblacion):
    """
    Para cada solucion de la poblacion calcula, en cada maquina:
      - cuanto tiempo estara trabajando
      - cuanta memoria usaran sus tareas
    poblacion es una tabla: una fila por solucion y una columna por tarea.
    """
    cantidad_de_soluciones = poblacion.shape[0]
    numero_de_maquinas = problema["numero_de_maquinas"]
    tiempo_ocupado = numpy.zeros((cantidad_de_soluciones, numero_de_maquinas))
    memoria_usada = numpy.zeros((cantidad_de_soluciones, numero_de_maquinas))

    for maquina in range(numero_de_maquinas):
        # Tabla de verdadero/falso: ¿esta tarea esta asignada a esta maquina?
        tarea_asignada_a_la_maquina = (poblacion == maquina)
        tiempo_de_cada_tarea_en_la_maquina = (problema["longitud_de_cada_tarea"] / problema["velocidad_de_cada_maquina"][maquina])
        # : solos es selecccionar todo
        tiempo_ocupado[:, maquina] = (tarea_asignada_a_la_maquina * tiempo_de_cada_tarea_en_la_maquina).sum(axis=1)
        memoria_usada[:, maquina] = (tarea_asignada_a_la_maquina * problema["memoria_de_cada_tarea"]).sum(axis=1)
    return tiempo_ocupado, memoria_usada


def calcular_metricas(problema, tiempo_ocupado, memoria_usada):
    #Calcula el tiempo total, la energia y cuanta memoria se excede

    # El tiempo total (makespan) es lo que tarda la maquina mas lenta en terminar
    tiempo_total = tiempo_ocupado.max(axis=1)

    # Energia: cada maquina gasta mas mientras trabaja y menos mientras espera a que todas terminen
    tiempo_en_reposo = tiempo_total[:, None] - tiempo_ocupado
    energia = (problema["potencia_en_uso"] * tiempo_ocupado + problema["potencia_en_reposo"] * tiempo_en_reposo).sum(axis=1)

    # Exceso de memoria: cuanto se pasa cada maquina de su limite (0 si no se pasa)
    memoria_que_sobra = numpy.maximum(0.0, memoria_usada - problema["capacidad_de_memoria"])
    exceso_de_memoria = (memoria_que_sobra / problema["capacidad_de_memoria"]).sum(axis=1)
    return tiempo_total, energia, exceso_de_memoria

def calcular_fitness_desde_metricas(problema, tiempo_total, energia, exceso_de_memoria):
    # El fitness es la calificacion de una solucion. MENOR ES MEJOR. Mezcla tiempo, energia y un castigo (penalizacion) por pasarse de memoria.

    parte_del_tiempo = problema["peso_del_tiempo"] * tiempo_total / problema["tiempo_de_referencia"]
    parte_de_la_energia = problema["peso_de_la_energia"] * energia / problema["energia_de_referencia"]
    castigo = problema["factor_de_penalizacion"] * exceso_de_memoria
    return parte_del_tiempo + parte_de_la_energia + castigo

def calcular_fitness(problema, poblacion):
    # Calcula el fitness de todas las soluciones de la poblacion
    tiempo_ocupado, memoria_usada = calcular_tiempo_y_memoria_por_maquina(problema, poblacion)
    tiempo_total, energia, exceso_de_memoria = calcular_metricas(problema, tiempo_ocupado, memoria_usada)
    return calcular_fitness_desde_metricas(problema, tiempo_total, energia, exceso_de_memoria)

def describir_solucion(problema, solucion):
    # Devuelve (fitness, tiempo total, energia, exceso de memoria) de UNA (1) solucion
    poblacion_de_una_solucion = numpy.array([solucion])
    tiempo_ocupado, memoria_usada = calcular_tiempo_y_memoria_por_maquina(problema, poblacion_de_una_solucion)
    tiempo_total, energia, exceso_de_memoria = calcular_metricas(problema, tiempo_ocupado, memoria_usada)
    fitness = calcular_fitness_desde_metricas(problema, tiempo_total, energia, exceso_de_memoria)
    return float(fitness[0]), float(tiempo_total[0]), float(energia[0]), float(exceso_de_memoria[0])


# =============================================================================
# METODOS DE REFERENCIA (para saber si las metaheuristicas sirven)
# =============================================================================

def crear_solucion_round_robin(problema):
    #Reparte las tareas por turnos: tarea 0 a maquina 0, tarea 1 a maquina 1, ...
    solucion = numpy.zeros(problema["numero_de_tareas"], dtype=int)
    for tarea in range(problema["numero_de_tareas"]):
        solucion[tarea] = tarea % problema["numero_de_maquinas"]
    return solucion

def crear_solucion_voraz(problema):
    # Metodo voraz: toma las tareas de la mas larga a la mas corta y manda cada una a la maquina donde el fitness quede mejor en ese momento.
    
    numero_de_maquinas = problema["numero_de_maquinas"]
    tareas_de_mayor_a_menor = numpy.argsort(-problema["longitud_de_cada_tarea"])
    solucion = numpy.zeros(problema["numero_de_tareas"], dtype=int)
    tiempo_ocupado = numpy.zeros((1, numero_de_maquinas))
    memoria_usada = numpy.zeros((1, numero_de_maquinas))

    for tarea in tareas_de_mayor_a_menor:
        mejor_maquina = 0
        mejor_fitness = numpy.inf
        for maquina in range(numero_de_maquinas):
            # Probamos que pasaria si la tarea se asigna a esta maquina
            tiempo_de_prueba = tiempo_ocupado.copy()
            memoria_de_prueba = memoria_usada.copy()
            tiempo_de_prueba[0, maquina] += problema["longitud_de_cada_tarea"][tarea] / problema["velocidad_de_cada_maquina"][maquina]
            memoria_de_prueba[0, maquina] += problema["memoria_de_cada_tarea"][tarea]
            tiempo_total, energia, exceso_de_memoria = calcular_metricas(problema, tiempo_de_prueba, memoria_de_prueba)
            fitness = calcular_fitness_desde_metricas(problema, tiempo_total, energia, exceso_de_memoria)[0]
            if fitness < mejor_fitness:
                mejor_fitness = fitness
                mejor_maquina = maquina
        # Nos quedamos con la mejor maquina encontrada
        solucion[tarea] = mejor_maquina
        tiempo_ocupado[0, mejor_maquina] += problema["longitud_de_cada_tarea"][tarea] / problema["velocidad_de_cada_maquina"][mejor_maquina]
        memoria_usada[0, mejor_maquina] += problema["memoria_de_cada_tarea"][tarea]
    return solucion

# =============================================================================
# LOS TRES ALGORITMOS METAHEURISTICOS
# =============================================================================
# Los tres devuelven:
#   1) La mejor solucion encontrada
#   2) El historial: el mejor fitness al final de cada iteracion (sirve para graficar)
#   3) Informacion extra (solo el hibrido la usa)

def seleccionar_ganadores_de_torneo(generador_aleatorio, fitness_de_la_poblacion, cantidad_de_torneos, participantes_por_torneo):
    #Seleccion por torneo: se eligen algunas soluciones al azar y gana la de menor fitness, devuelve la posicion de cada ganador dentro de la poblacion
    
    cantidad_de_soluciones = len(fitness_de_la_poblacion)
    competidores_de_cada_torneo = generador_aleatorio.integers(0, cantidad_de_soluciones, (cantidad_de_torneos, participantes_por_torneo))
    ganadores = numpy.zeros(cantidad_de_torneos, dtype=int)
    for numero_de_torneo in range(cantidad_de_torneos):
        competidores = competidores_de_cada_torneo[numero_de_torneo]
        posicion_del_mejor = fitness_de_la_poblacion[competidores].argmin()
        ganadores[numero_de_torneo] = competidores[posicion_del_mejor]
    return ganadores

# ---------------------------------------------------------------- ALGORITMO GENETICO
def algoritmo_genetico(problema, generador_aleatorio, tamano_de_poblacion=60, numero_de_iteraciones=300, probabilidad_de_cruce=0.9, probabilidad_de_mutacion=0.005, cantidad_de_elite=2, participantes_por_torneo=3):
    """
    Algoritmo genetico:
      1. Se crea una poblacion de soluciones al azar
      2. En cada generacion: se eligen padres (torneo), se cruzan y se mutan
      3. Los mejores padres pasan directo a la siguiente generacion
    """
    numero_de_tareas = problema["numero_de_tareas"]
    numero_de_maquinas = problema["numero_de_maquinas"]

    poblacion = generador_aleatorio.integers(0, numero_de_maquinas, (tamano_de_poblacion, numero_de_tareas))
    fitness_de_la_poblacion = calcular_fitness(problema, poblacion)
    historial_del_mejor_fitness = [fitness_de_la_poblacion.min()]

    for iteracion in range(numero_de_iteraciones):
        # Seleccion: se eligen dos grupos de padres
        ganadores = seleccionar_ganadores_de_torneo(generador_aleatorio, fitness_de_la_poblacion, 2 * tamano_de_poblacion, participantes_por_torneo)
        primeros_padres = poblacion[ganadores[:tamano_de_poblacion]]
        segundos_padres = poblacion[ganadores[tamano_de_poblacion:]]

        # Cruce uniforme: cada gen del hijo se copia del primer o del segundo padre (50 % y 50 %)
        gen_viene_del_segundo_padre = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) < 0.5
        este_hijo_se_cruza = generador_aleatorio.random(tamano_de_poblacion) < probabilidad_de_cruce
        usar_segundo_padre = gen_viene_del_segundo_padre & este_hijo_se_cruza[:, None]
        hijos = numpy.where(usar_segundo_padre, segundos_padres, primeros_padres)

        # Mutacion: algunos genes cambian a una maquina al azar
        gen_muta = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) < probabilidad_de_mutacion
        maquinas_al_azar = generador_aleatorio.integers(0, numero_de_maquinas, (tamano_de_poblacion, numero_de_tareas))
        hijos = numpy.where(gen_muta, maquinas_al_azar, hijos)
        fitness_de_los_hijos = calcular_fitness(problema, hijos)

        # Elitismo: los mejores padres reemplazan a los peores hijos
        posiciones_de_los_mejores_padres = numpy.argsort(fitness_de_la_poblacion)[:cantidad_de_elite]
        posiciones_de_los_peores_hijos = numpy.argsort(fitness_de_los_hijos)[-cantidad_de_elite:]
        hijos[posiciones_de_los_peores_hijos] = poblacion[posiciones_de_los_mejores_padres]
        fitness_de_los_hijos[posiciones_de_los_peores_hijos] = fitness_de_la_poblacion[posiciones_de_los_mejores_padres]

        poblacion = hijos
        fitness_de_la_poblacion = fitness_de_los_hijos
        historial_del_mejor_fitness.append(fitness_de_la_poblacion.min())

    posicion_del_mejor = fitness_de_la_poblacion.argmin()
    return poblacion[posicion_del_mejor].copy(), numpy.array(historial_del_mejor_fitness), {}

# ---------------------------------------------------------------- PSO
def optimizacion_por_enjambre_de_particulas(problema, generador_aleatorio, tamano_de_poblacion=60, numero_de_iteraciones=300, inercia_inicial=0.7, inercia_final=0.3, coeficiente_cognitivo=2.0, coeficiente_social=2.0):
    """
    PSO (enjambre de particulas):
      Cada particula tiene una posicion (un numero por tarea entre 0 y la cantidad de maquinas) y una velocidad. En cada paso se mueve un poco hacia:
        - la mejor posicion que ella misma ha encontrado
        - la mejor posicion que ha encontrado todo el enjambre
      Para evaluar, la posicion se convierte en numero entero (se quitan los decimales)
    """
    numero_de_tareas = problema["numero_de_tareas"]
    numero_de_maquinas = problema["numero_de_maquinas"]
    velocidad_maxima = 0.25 * numero_de_maquinas

    posiciones = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) * numero_de_maquinas
    velocidades = (generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) - 0.5) * velocidad_maxima
    soluciones = numpy.clip(posiciones.astype(int), 0, numero_de_maquinas - 1)
    fitness_actual = calcular_fitness(problema, soluciones)

    mejores_posiciones_personales = posiciones.copy()
    mejores_fitness_personales = fitness_actual.copy()
    posicion_del_mejor_global = mejores_fitness_personales.argmin()
    mejor_posicion_global = mejores_posiciones_personales[posicion_del_mejor_global].copy()
    mejor_fitness_global = mejores_fitness_personales[posicion_del_mejor_global]
    historial_del_mejor_fitness = [mejor_fitness_global]

    for iteracion in range(numero_de_iteraciones):
        # La inercia baja poco a poco: al inicio explora mucho, al final ajusta con mas cuidado
        inercia = inercia_inicial - (inercia_inicial - inercia_final) * iteracion / numero_de_iteraciones
        azar_cognitivo = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas))
        azar_social = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas))

        velocidades = (inercia * velocidades + coeficiente_cognitivo * azar_cognitivo * (mejores_posiciones_personales - posiciones) + coeficiente_social * azar_social * (mejor_posicion_global - posiciones))
        velocidades = numpy.clip(velocidades, -velocidad_maxima, velocidad_maxima)
        posiciones = numpy.clip(posiciones + velocidades, 0, numero_de_maquinas - 1e-9)

        soluciones = numpy.clip(posiciones.astype(int), 0, numero_de_maquinas - 1)
        fitness_actual = calcular_fitness(problema, soluciones)

        # Si una particula mejoro, se actualiza su mejor posicion personal
        mejoraron = fitness_actual < mejores_fitness_personales
        mejores_posiciones_personales[mejoraron] = posiciones[mejoraron]
        mejores_fitness_personales[mejoraron] = fitness_actual[mejoraron]

        # Si alguna supero al mejor global, se actualiza el mejor global
        posicion_del_mejor_actual = mejores_fitness_personales.argmin()
        if mejores_fitness_personales[posicion_del_mejor_actual] < mejor_fitness_global:
            mejor_fitness_global = mejores_fitness_personales[posicion_del_mejor_actual]
            mejor_posicion_global = mejores_posiciones_personales[posicion_del_mejor_actual].copy()
        historial_del_mejor_fitness.append(mejor_fitness_global)

    mejor_solucion = numpy.clip(mejor_posicion_global.astype(int), 0, numero_de_maquinas - 1)
    return mejor_solucion, numpy.array(historial_del_mejor_fitness), {}


# ---------------------------------------------------------------- HIBRIDO GA-PSO
def algoritmo_hibrido(problema, generador_aleatorio, tamano_de_poblacion=60, numero_de_iteraciones=300, probabilidad_de_cruce=0.9, mutacion_minima=0.002, mutacion_maxima=0.02, intensidad_de_atraccion=0.3, inercia_inicial=0.9, inercia_final=0.4, coeficiente_cognitivo=1.5, coeficiente_social=1.5, participantes_por_torneo=3):
    """
    Hibrido GA-PSO. Es un algoritmo genetico al que se le agregan ideas del PSO:
      1. Poblacion inicial genetica (soluciones al azar)
      2. Cruce y mutacion, igual que en el algoritmo genetico
      3. Ajuste dinamico de parametros (idea del enjambre):
           - Si las soluciones se parecen demasiado entre si (poca diversidad), sube la mutacion.
           - Cada gen del hijo puede ser "jalado" hacia la mejor solucion del individuo o hacia la mejor solucion de toda la poblacion.
      4. Reemplazo elitista: un hijo sustituye a su padre SOLO si tiene mejor fitness.
      5. Se guarda siempre el mejor resultado global.
    """
    numero_de_tareas = problema["numero_de_tareas"]
    numero_de_maquinas = problema["numero_de_maquinas"]

    poblacion = generador_aleatorio.integers(0, numero_de_maquinas, (tamano_de_poblacion, numero_de_tareas))
    fitness_de_la_poblacion = calcular_fitness(problema, poblacion)
    posicion_del_mejor = fitness_de_la_poblacion.argmin()
    mejor_solucion_global = poblacion[posicion_del_mejor].copy()
    mejor_fitness_global = fitness_de_la_poblacion[posicion_del_mejor]

    # Diversidad que tiene una poblacion totalmente aleatoria (sirve como referencia)
    diversidad_inicial_esperada = (numero_de_maquinas - 1) / numero_de_maquinas
    probabilidad_de_jalar_hacia_el_mejor_personal = coeficiente_cognitivo / (coeficiente_cognitivo + coeficiente_social)

    historial_del_mejor_fitness = [mejor_fitness_global]
    registro = {"mutacion": [], "diversidad": [], "inercia": []}

    for iteracion in range(numero_de_iteraciones):
        inercia = inercia_inicial - (inercia_inicial - inercia_final) * iteracion / numero_de_iteraciones

        # Diversidad: que tan distintas son las soluciones respecto a la mejor (0 = todas iguales)
        diversidad = float((poblacion != mejor_solucion_global).mean())
        nivel_de_convergencia = 1.0 - diversidad / diversidad_inicial_esperada
        nivel_de_convergencia = min(1.0, max(0.0, nivel_de_convergencia))
        # Mientras mas parecidas son las soluciones, mas sube la mutacion
        probabilidad_de_mutacion = mutacion_minima + (mutacion_maxima - mutacion_minima) * nivel_de_convergencia

        # Parte genetica: seleccion, cruce y mutacion
        ganadores = seleccionar_ganadores_de_torneo(generador_aleatorio, fitness_de_la_poblacion,
                                                    2 * tamano_de_poblacion, participantes_por_torneo)
        primeros_padres = poblacion[ganadores[:tamano_de_poblacion]]
        segundos_padres = poblacion[ganadores[tamano_de_poblacion:]]
        gen_viene_del_segundo_padre = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) < 0.5
        este_hijo_se_cruza = generador_aleatorio.random(tamano_de_poblacion) < probabilidad_de_cruce
        usar_segundo_padre = gen_viene_del_segundo_padre & este_hijo_se_cruza[:, None]
        hijos = numpy.where(usar_segundo_padre, segundos_padres, primeros_padres)

        gen_muta = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) < probabilidad_de_mutacion
        maquinas_al_azar = generador_aleatorio.integers(0, numero_de_maquinas, (tamano_de_poblacion, numero_de_tareas))
        hijos = numpy.where(gen_muta, maquinas_al_azar, hijos)

        # Parte de enjambre: algunos genes se "jalan" hacia las mejores soluciones
        probabilidad_de_jalar = intensidad_de_atraccion * (1.0 - inercia)
        gen_es_jalado = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) < probabilidad_de_jalar
        jalar_hacia_el_mejor_personal = generador_aleatorio.random((tamano_de_poblacion, numero_de_tareas)) < probabilidad_de_jalar_hacia_el_mejor_personal
        # El "mejor personal" de cada individuo es el propio individuo (por el reemplazo elitista)
        hijos = numpy.where(gen_es_jalado & jalar_hacia_el_mejor_personal, poblacion, hijos)
        hijos = numpy.where(gen_es_jalado & ~jalar_hacia_el_mejor_personal, mejor_solucion_global[None, :], hijos)

        # Reemplazo elitista: el hijo solo reemplaza a su individuo si es mejor
        fitness_de_los_hijos = calcular_fitness(problema, hijos)
        hijo_es_mejor = fitness_de_los_hijos < fitness_de_la_poblacion
        poblacion[hijo_es_mejor] = hijos[hijo_es_mejor]
        fitness_de_la_poblacion[hijo_es_mejor] = fitness_de_los_hijos[hijo_es_mejor]

        # Se actualiza el mejor global
        posicion_del_mejor = fitness_de_la_poblacion.argmin()
        if fitness_de_la_poblacion[posicion_del_mejor] < mejor_fitness_global:
            mejor_fitness_global = fitness_de_la_poblacion[posicion_del_mejor]
            mejor_solucion_global = poblacion[posicion_del_mejor].copy()
        historial_del_mejor_fitness.append(mejor_fitness_global)

        registro["mutacion"].append(probabilidad_de_mutacion)
        registro["diversidad"].append(diversidad)
        registro["inercia"].append(inercia)

    return mejor_solucion_global, numpy.array(historial_del_mejor_fitness), registro

# =============================================================================
# REPETIR LOS EXPERIMENTOS
# =============================================================================
# Los algoritmos usan numeros al azar, asi que cada corrida da un resultado distinto
# Por eso cada algoritmo se ejecuta muchas veces y se analizan los resultados juntos

def ejecutar_varias_corridas(problema, nombre_del_algoritmo, numero_de_corridas, tamano_de_poblacion, numero_de_iteraciones, semilla_inicial, probabilidad_de_cruce=0.9, mutacion_maxima=0.02, intensidad_de_atraccion=0.3):
    """
    Ejecuta un algoritmo varias veces. Cada corrida usa una semilla distinta pero siempre
    la misma lista de semillas, para que los resultados se puedan repetir.
    Los ultimos tres parametros solo los usa el hibrido
    """
    fitness_finales = []
    historiales = []
    tiempos_de_ejecucion = []
    mejores_soluciones = []

    for numero_de_corrida in range(numero_de_corridas):
        generador_aleatorio = numpy.random.default_rng(semilla_inicial + numero_de_corrida)
        hora_de_inicio = time.perf_counter()

        if nombre_del_algoritmo == "GA":
            solucion, historial, registro = algoritmo_genetico(problema, generador_aleatorio, tamano_de_poblacion, numero_de_iteraciones)
        elif nombre_del_algoritmo == "PSO":
            solucion, historial, registro = optimizacion_por_enjambre_de_particulas(problema, generador_aleatorio, tamano_de_poblacion, numero_de_iteraciones)
        else:  # GA-PSO
            solucion, historial, registro = algoritmo_hibrido(problema, generador_aleatorio, tamano_de_poblacion, numero_de_iteraciones, probabilidad_de_cruce=probabilidad_de_cruce, mutacion_maxima=mutacion_maxima, intensidad_de_atraccion=intensidad_de_atraccion)

        tiempos_de_ejecucion.append(time.perf_counter() - hora_de_inicio)
        fitness_finales.append(historial[-1])
        historiales.append(historial)
        mejores_soluciones.append(solucion)

    return {"fitness_final": numpy.array(fitness_finales),
            "historial": numpy.array(historiales),
            "tiempo": numpy.array(tiempos_de_ejecucion),
            "soluciones": numpy.array(mejores_soluciones)}

def contar_iteraciones_hasta_umbral(historiales, umbral):
    #Para cada corrida, indica en que iteracion se llego a un fitness menor o igual al umbral (nan si nunca)
    iteraciones = []
    for historial in historiales:
        iteracion_encontrada = numpy.nan
        for numero_de_iteracion in range(len(historial)):
            if historial[numero_de_iteracion] <= umbral:
                iteracion_encontrada = numero_de_iteracion
                break
        iteraciones.append(iteracion_encontrada)
    return numpy.array(iteraciones, dtype=float)

def calcular_carga_por_maquina(problema, solucion):
    #Tiempo que trabaja cada maquina con una solucion dada (sirve para la grafica de balance de carga)
    tiempo_ocupado, memoria_usada = calcular_tiempo_y_memoria_por_maquina(problema, numpy.array([solucion]))
    return tiempo_ocupado[0]

# =============================================================================
# GRAFICAS
# =============================================================================
COLORES = {"GA": "#1f77b4", "PSO": "#d62728", "GA-PSO": "#2ca02c", "Voraz": "#7f7f7f", "Round-Robin": "#ff7f0e"}
pyplot.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130, "savefig.bbox": "tight"})

def graficar_convergencia(resultados, fitness_del_voraz, numero_de_iteraciones, carpeta):
    #Figura 1: como mejora el fitness a lo largo de las iteraciones
    iteraciones = numpy.arange(numero_de_iteraciones + 1)
    figura, ejes = pyplot.subplots(1, 2, figsize=(9, 3.4))
    for nombre in resultados:
        promedio = resultados[nombre]["historial"].mean(axis=0)
        desviacion_estandar = resultados[nombre]["historial"].std(axis=0)
        for numero_de_eje in (0, 1):
            if numero_de_eje == 1 and nombre == "PSO":
                continue  # En el detalle final el PSO queda fuera de la escala
            ejes[numero_de_eje].plot(iteraciones, promedio, color=COLORES[nombre], label=nombre, linewidth=1.4)
            ejes[numero_de_eje].fill_between(iteraciones, promedio - desviacion_estandar, promedio + desviacion_estandar, color=COLORES[nombre], alpha=0.18)
    for numero_de_eje in (0, 1):
        ejes[numero_de_eje].axhline(fitness_del_voraz, color=COLORES["Voraz"], linestyle="--", linewidth=1, label="Voraz (baseline)")
        ejes[numero_de_eje].set_xlabel("Iteracion")
        ejes[numero_de_eje].set_ylabel("Mejor fitness (↓)")
    ejes[0].set_title("(a) Convergencia completa")
    ejes[1].set_title("(b) Detalle final")
    ejes[1].set_ylim(fitness_del_voraz - 0.008, fitness_del_voraz + 0.03)
    ejes[1].set_xlim(50, numero_de_iteraciones)
    ejes[0].set_yscale("log")
    ejes[0].legend()
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig1_convergencia.png"))
    pyplot.close(figura)

def graficar_cajas(resultados, fitness_del_voraz, carpeta):
    #Figura 2: diagrama de cajas con el fitness final de todas las corridas
    figura, eje = pyplot.subplots(figsize=(4.6, 3.4))
    nombres = list(resultados.keys())
    datos = []
    for nombre in nombres:
        datos.append(resultados[nombre]["fitness_final"])
    cajas = eje.boxplot(datos, tick_labels=nombres, patch_artist=True)
    for caja, nombre in zip(cajas["boxes"], nombres):
        caja.set_facecolor(COLORES[nombre])
        caja.set_alpha(0.55)
    eje.axhline(fitness_del_voraz, color=COLORES["Voraz"], linestyle="--", linewidth=1, label="Voraz")
    eje.set_ylabel("Fitness final (↓)")
    eje.set_title("Robustez: 30 corridas independientes")
    eje.legend()
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig2_boxplot.png"))
    pyplot.close(figura)

def graficar_tiempo_y_energia(nombres, tiempos_totales, energias, carpeta):
    #Figura 3: barras con el tiempo total y la energia de cada metodo
    colores_de_las_barras = []
    for nombre in nombres:
        colores_de_las_barras.append(COLORES[nombre])
    energias_en_kilojoules = numpy.array(energias) / 1000

    figura, ejes = pyplot.subplots(1, 2, figsize=(9, 3.2))
    ejes[0].bar(nombres, tiempos_totales, color=colores_de_las_barras)
    ejes[0].set_ylabel("Makespan (s)")
    ejes[0].set_title("(a) Tiempo de respuesta")
    ejes[1].bar(nombres, energias_en_kilojoules, color=colores_de_las_barras)
    ejes[1].set_ylabel("Energia (kJ)")
    ejes[1].set_title("(b) Consumo energetico")
    for eje in ejes:
        eje.tick_params(axis="x", rotation=25)
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig3_makespan_energia.png"))
    pyplot.close(figura)

def graficar_asignacion(problema, solucion_round_robin, solucion_voraz, mejor_solucion_hibrida, carpeta):
    #Figura 4: carga de trabajo por maquina y mapa de que tarea va a que maquina
    numero_de_maquinas = problema["numero_de_maquinas"]
    numero_de_tareas = problema["numero_de_tareas"]
    figura, ejes = pyplot.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})

    posiciones_en_x = numpy.arange(numero_de_maquinas)
    ancho_de_barra = 0.27
    ejes[0].bar(posiciones_en_x - ancho_de_barra, calcular_carga_por_maquina(problema, solucion_round_robin), ancho_de_barra, color=COLORES["Round-Robin"], label="Round-Robin")
    ejes[0].bar(posiciones_en_x, calcular_carga_por_maquina(problema, solucion_voraz), ancho_de_barra, color=COLORES["Voraz"], label="Voraz")
    ejes[0].bar(posiciones_en_x + ancho_de_barra, calcular_carga_por_maquina(problema, mejor_solucion_hibrida), ancho_de_barra, color=COLORES["GA-PSO"], label="GA-PSO")
    ejes[0].set_xticks(posiciones_en_x)
    ejes[0].set_xlabel("VM (ordenadas por indice)")
    ejes[0].set_ylabel("Tiempo ocupado (s)")
    ejes[0].set_title("(a) Balance de carga por VM")
    ejes[0].legend()

    # Mapa: una fila por maquina (de la mas lenta a la mas rapida) y una columna por tarea
    maquinas_de_lenta_a_rapida = numpy.argsort(problema["velocidad_de_cada_maquina"])
    mapa = numpy.zeros((numero_de_maquinas, numero_de_tareas))
    for tarea in range(numero_de_tareas):
        maquina_asignada = mejor_solucion_hibrida[tarea]
        fila_del_mapa = numpy.where(maquinas_de_lenta_a_rapida == maquina_asignada)[0][0]
        mapa[fila_del_mapa, tarea] = problema["longitud_de_cada_tarea"][tarea]
    imagen = ejes[1].imshow(mapa, aspect="auto", cmap="viridis", interpolation="nearest")
    ejes[1].set_yticks(range(numero_de_maquinas))
    etiquetas_de_velocidad = []
    for maquina in maquinas_de_lenta_a_rapida:
        etiquetas_de_velocidad.append(str(int(problema["velocidad_de_cada_maquina"][maquina])))
    ejes[1].set_yticklabels(etiquetas_de_velocidad)
    ejes[1].set_xlabel("Tarea")
    ejes[1].set_ylabel("VM (MIPS)")
    ejes[1].set_title("(b) Mapa de asignacion (color = MI)")
    ejes[1].grid(False)
    figura.colorbar(imagen, ax=ejes[1], fraction=0.04)
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig4_asignacion.png"))
    pyplot.close(figura)

def graficar_parametros_adaptativos(registro, carpeta):
    # Figura 5: como cambian la diversidad y la mutacion durante una corrida del hibrido
    figura, eje_izquierdo = pyplot.subplots(figsize=(4.8, 3.2))
    eje_izquierdo.plot(registro["diversidad"], color="#9467bd")
    eje_izquierdo.set_xlabel("Iteracion")
    eje_izquierdo.set_ylabel("Diversidad", color="#9467bd")
    eje_derecho = eje_izquierdo.twinx()
    eje_derecho.plot(registro["mutacion"], color="#e377c2")
    eje_derecho.set_ylabel("$p_m$", color="#e377c2")
    eje_derecho.grid(False)
    eje_izquierdo.set_title("Ajuste dinamico de parametros (GA-PSO)")
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig5_parametros_adaptativos.png"))
    pyplot.close(figura)

def graficar_sensibilidad(sensibilidad, carpeta):
    # Figura 6: que tanto cambia el resultado al modificar cada parametro del hibrido
    titulos = {"tamano_poblacion": "Tamaño de poblacion", "pull_enjambre": "Atraccion de enjambre", "pm_max": "$p_m$ maxima", "pc": "$p_c$ (cruce)"}
    figura, ejes = pyplot.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    numero_de_eje = 0
    for nombre_del_parametro in sensibilidad:
        valores_probados = list(sensibilidad[nombre_del_parametro].keys())
        promedios = []
        desviaciones = []
        for valor in valores_probados:
            promedios.append(sensibilidad[nombre_del_parametro][valor][0])
            desviaciones.append(sensibilidad[nombre_del_parametro][valor][1])
        eje = ejes[numero_de_eje]
        eje.errorbar(range(len(valores_probados)), promedios, yerr=desviaciones, marker="o", color=COLORES["GA-PSO"], capsize=3)
        eje.set_xticks(range(len(valores_probados)))
        eje.set_xticklabels([str(valor) for valor in valores_probados])
        eje.set_title(titulos[nombre_del_parametro])
        numero_de_eje += 1
    ejes[0].set_ylabel("Fitness final (↓)")
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig6_sensibilidad.png"))
    pyplot.close(figura)

def graficar_compromiso(lista_de_pesos, carpeta):
    # Figura 7: relacion entre tiempo total y energia al cambiar los pesos del objetivo
    figura, eje = pyplot.subplots(figsize=(4.6, 3.3))
    tiempos = []
    energias_en_kilojoules = []
    for punto in lista_de_pesos:
        tiempos.append(punto["makespan"])
        energias_en_kilojoules.append(punto["energia"] / 1000)
    eje.plot(tiempos, energias_en_kilojoules, "o-", color=COLORES["GA-PSO"])
    for punto in lista_de_pesos:
        eje.annotate(f"w={punto['w_time']}", (punto["makespan"], punto["energia"] / 1000), textcoords="offset points", xytext=(5, 4), fontsize=7)
    eje.set_xlabel("Makespan (s)")
    eje.set_ylabel("Energia (kJ)")
    eje.set_title("Compromiso tiempo vs. energia")
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig7_compromiso.png"))
    pyplot.close(figura)

def graficar_escalabilidad(resultados_por_tamano, carpeta):
    #Figura 8: fitness de cada algoritmo comparado con el del voraz en problemas de distinto tamaño
    figura, eje = pyplot.subplots(figsize=(5.2, 3.2))
    etiquetas = []
    for fila in resultados_por_tamano:
        etiquetas.append(f"{fila['n']}×{fila['m']}")
    posiciones_en_x = numpy.arange(len(resultados_por_tamano))
    ancho_de_barra = 0.26
    eje.set_ylim(0.9, 1.6)

    numero_de_algoritmo = 0
    for nombre in ["GA", "PSO", "GA-PSO"]:
        valores_relativos = []
        for fila in resultados_por_tamano:
            valores_relativos.append(fila[nombre]["media"] / fila["Voraz"])
        desplazamiento = (numero_de_algoritmo - 1) * ancho_de_barra
        eje.bar(posiciones_en_x + desplazamiento, valores_relativos, ancho_de_barra, color=COLORES[nombre], label=nombre)
        # Las barras que no caben en la grafica llevan su valor escrito encima
        for posicion, valor in zip(posiciones_en_x, valores_relativos):
            if valor > 1.6:
                eje.text(posicion + desplazamiento, 1.57, f"{valor:.1f}×", ha="center", va="top", fontsize=7, color="white", rotation=90)
        numero_de_algoritmo += 1

    eje.axhline(1, color=COLORES["Voraz"], linestyle="--", linewidth=1, label="Voraz = 1")
    eje.set_xticks(posiciones_en_x)
    eje.set_xticklabels(etiquetas)
    eje.set_xlabel("Tareas × VMs")
    eje.set_ylabel("Fitness / fitness Voraz (↓)")
    eje.legend(fontsize=7, loc="upper left")
    eje.set_title("Escalabilidad")
    figura.tight_layout()
    figura.savefig(os.path.join(carpeta, "fig8_escalabilidad.png"))
    pyplot.close(figura)

# =============================================================================
# EXPERIMENTO COMPLETO Y ESTADISTICA
# =============================================================================

def comparar_con_wilcoxon(valores_del_primer_algoritmo, valores_del_segundo_algoritmo):
    # Prueba de Wilcoxon. Devuelve el valor p (si es menor que 0.05, la diferencia es significativa)
    try:
        return float(stats.wilcoxon(valores_del_primer_algoritmo, valores_del_segundo_algoritmo).pvalue)
    except ValueError:
        return 1.0

def promedio_y_desviacion_del_hibrido(problema, numero_de_corridas, evaluaciones_disponibles, tamano_de_poblacion, probabilidad_de_cruce, mutacion_maxima, intensidad_de_atraccion):
    """Corre el hibrido varias veces con ciertos parametros y devuelve (promedio, desviacion estandar)."""
    # El numero de iteraciones se ajusta para que todos usen la misma cantidad de evaluaciones
    numero_de_iteraciones = evaluaciones_disponibles // tamano_de_poblacion
    datos = ejecutar_varias_corridas(problema, "GA-PSO", numero_de_corridas, tamano_de_poblacion, numero_de_iteraciones, semilla_inicial=7000, probabilidad_de_cruce=probabilidad_de_cruce, mutacion_maxima=mutacion_maxima, intensidad_de_atraccion=intensidad_de_atraccion)
    return float(datos["fitness_final"].mean()), float(datos["fitness_final"].std(ddof=1))

def ejecutar_experimento_completo(modo_rapido=False, carpeta_de_salida="resultados"):
    os.makedirs(carpeta_de_salida, exist_ok=True)

    # Configuracion: el modo rapido usa menos corridas e iteraciones
    if modo_rapido:
        numero_de_corridas, numero_de_iteraciones, tamano_de_poblacion = 8, 150, 60
        corridas_de_sensibilidad, evaluaciones_de_sensibilidad = 4, 9000
        corridas_de_compromiso, corridas_de_escalabilidad = 3, 4
    else:
        numero_de_corridas, numero_de_iteraciones, tamano_de_poblacion = 30, 500, 60
        corridas_de_sensibilidad, evaluaciones_de_sensibilidad = 15, 30000
        corridas_de_compromiso, corridas_de_escalabilidad = 10, 15

    nombres_de_algoritmos = ["GA", "PSO", "GA-PSO"]
    problema = crear_problema(100, 12)
    informe = {"config": {"runs": numero_de_corridas, "iters": numero_de_iteraciones, "pop": tamano_de_poblacion, "n_tareas": problema["numero_de_tareas"], "n_vms": problema["numero_de_maquinas"], "evaluaciones": tamano_de_poblacion * (numero_de_iteraciones + 1), "w_time": problema["peso_del_tiempo"], "penalty": problema["factor_de_penalizacion"]}}

    # ------------------------------------------------ 1. Comparacion principal
    print("1/4 Comparacion principal...")
    resultados = {}
    for nombre in nombres_de_algoritmos:
        resultados[nombre] = ejecutar_varias_corridas(problema, nombre, numero_de_corridas, tamano_de_poblacion, numero_de_iteraciones, semilla_inicial=1000)

    solucion_round_robin = crear_solucion_round_robin(problema)
    solucion_voraz = crear_solucion_voraz(problema)
    informe["base"] = {}
    for nombre, solucion in [("Round-Robin", solucion_round_robin), ("Voraz", solucion_voraz)]:
        fitness, tiempo_total, energia, exceso = describir_solucion(problema, solucion)
        informe["base"][nombre] = {"fitness": fitness, "makespan": tiempo_total, "energia": energia, "violacion": exceso}
    fitness_del_voraz = informe["base"]["Voraz"]["fitness"]

    # Umbral de convergencia: 1 % por encima del mejor valor encontrado por cualquier algoritmo
    mejor_fitness_de_todos = min(resultados[nombre]["fitness_final"].min() for nombre in nombres_de_algoritmos)
    umbral_de_convergencia = 1.01 * mejor_fitness_de_todos
    informe["umbral_convergencia"] = float(umbral_de_convergencia)

    informe["algoritmos"] = {}
    for nombre in nombres_de_algoritmos:
        datos = resultados[nombre]
        # Tiempo total y energia de la solucion final de cada corrida
        tiempos_totales = []
        energias = []
        excesos = []
        for solucion in datos["soluciones"]:
            fitness, tiempo_total, energia, exceso = describir_solucion(problema, solucion)
            tiempos_totales.append(tiempo_total)
            energias.append(energia)
            excesos.append(exceso)

        iteraciones_hasta_umbral = contar_iteraciones_hasta_umbral(datos["historial"], umbral_de_convergencia)
        corridas_que_llegaron = int(numpy.isfinite(iteraciones_hasta_umbral).sum())
        if corridas_que_llegaron > 0:
            promedio_de_iteraciones = float(numpy.nanmean(iteraciones_hasta_umbral))
        else:
            promedio_de_iteraciones = None

        # Iteracion en que se logro la mitad de toda la mejora (promedio de las corridas)
        iteraciones_de_la_mitad = []
        for historial in datos["historial"]:
            mitad_de_la_mejora = historial[0] - 0.5 * (historial[0] - historial[-1])
            iteraciones_de_la_mitad.append(numpy.argmax(historial <= mitad_de_la_mejora))

        fitness_finales = datos["fitness_final"]
        informe["algoritmos"][nombre] = {
            "media": float(fitness_finales.mean()),
            "std": float(fitness_finales.std(ddof=1)),
            "mejor": float(fitness_finales.min()),
            "peor": float(fitness_finales.max()),
            "mediana": float(numpy.median(fitness_finales)),
            "makespan_medio": float(numpy.mean(tiempos_totales)),
            "energia_media": float(numpy.mean(energias)),
            "violacion_media": float(numpy.mean(excesos)),
            "tiempo_medio_s": float(datos["tiempo"].mean()),
            "iter_umbral_media": promedio_de_iteraciones,
            "corridas_que_alcanzan_umbral": corridas_que_llegaron,
            "iter_mitad_mejora": float(numpy.mean(iteraciones_de_la_mitad)),
            "mejora_vs_voraz_pct": float(100 * (fitness_del_voraz - fitness_finales.mean()) / fitness_del_voraz),
            "corridas_mejor_que_voraz": int((fitness_finales < fitness_del_voraz - 1e-12).sum()),
        }

    # Pruebas estadisticas
    informe["wilcoxon"] = {
        "GA-PSO vs GA": comparar_con_wilcoxon(resultados["GA-PSO"]["fitness_final"], resultados["GA"]["fitness_final"]),
        "GA-PSO vs PSO": comparar_con_wilcoxon(resultados["GA-PSO"]["fitness_final"], resultados["PSO"]["fitness_final"]),
        "GA vs PSO": comparar_con_wilcoxon(resultados["GA"]["fitness_final"], resultados["PSO"]["fitness_final"]),
    }
    anova = stats.f_oneway(resultados["GA"]["fitness_final"], resultados["PSO"]["fitness_final"], resultados["GA-PSO"]["fitness_final"])
    informe["anova"] = {"F": float(anova.statistic), "p": float(anova.pvalue)}
    kruskal = stats.kruskal(resultados["GA"]["fitness_final"], resultados["PSO"]["fitness_final"], resultados["GA-PSO"]["fitness_final"])
    informe["kruskal"] = {"H": float(kruskal.statistic), "p": float(kruskal.pvalue)}

    # Tabla en formato CSV
    with open(os.path.join(carpeta_de_salida, "tabla_resultados.csv"), "w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["algoritmo", "media", "desviacion_estandar", "mejor", "peor", "makespan", "energia_J", "tiempo_s"])
        for nombre in nombres_de_algoritmos:
            fila = informe["algoritmos"][nombre]
            escritor.writerow([nombre, fila["media"], fila["std"], fila["mejor"], fila["peor"], fila["makespan_medio"], fila["energia_media"], fila["tiempo_medio_s"]])

    # ------------------------------------------------ 2. Sensibilidad de parametros del hibrido
    print("2/4 Sensibilidad de parametros...")

    sensibilidad = {"tamano_poblacion": {}, "pull_enjambre": {}, "pm_max": {}, "pc": {}}
    for tamano in [20, 40, 60, 100, 200]:
        sensibilidad["tamano_poblacion"][tamano] = promedio_y_desviacion_del_hibrido(problema, corridas_de_sensibilidad, evaluaciones_de_sensibilidad, tamano, 0.9, 0.02, 0.3)
    for atraccion in [0.0, 0.15, 0.3, 0.6, 1.0]:
        sensibilidad["pull_enjambre"][atraccion] = promedio_y_desviacion_del_hibrido(problema, corridas_de_sensibilidad, evaluaciones_de_sensibilidad, 60, 0.9, 0.02, atraccion)
    for mutacion_maxima in [0.005, 0.01, 0.02, 0.05, 0.1]:
        sensibilidad["pm_max"][mutacion_maxima] = promedio_y_desviacion_del_hibrido(problema, corridas_de_sensibilidad, evaluaciones_de_sensibilidad, 60, 0.9, mutacion_maxima, 0.3)
    for cruce in [0.5, 0.7, 0.9, 1.0]:
        sensibilidad["pc"][cruce] = promedio_y_desviacion_del_hibrido(problema, corridas_de_sensibilidad, evaluaciones_de_sensibilidad, 60, cruce, 0.02, 0.3)
    informe["sensibilidad"] = {}
    for nombre_del_parametro in sensibilidad:
        informe["sensibilidad"][nombre_del_parametro] = {}
        for valor in sensibilidad[nombre_del_parametro]:
            informe["sensibilidad"][nombre_del_parametro][str(valor)] = sensibilidad[nombre_del_parametro][valor]

    # Compromiso entre tiempo y energia: se cambia el peso del tiempo en el objetivo
    compromiso = []
    for peso_del_tiempo in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        problema_con_otros_pesos = crear_problema(100, 12, peso_del_tiempo=peso_del_tiempo)
        datos = ejecutar_varias_corridas(problema_con_otros_pesos, "GA-PSO", corridas_de_compromiso, tamano_de_poblacion, numero_de_iteraciones, semilla_inicial=9000)
        tiempos_totales = []
        energias = []
        excesos = []
        for solucion in datos["soluciones"]:
            fitness, tiempo_total, energia, exceso = describir_solucion(problema_con_otros_pesos, solucion)
            tiempos_totales.append(tiempo_total)
            energias.append(energia)
            excesos.append(exceso)
        compromiso.append({"w_time": peso_del_tiempo, "makespan": float(numpy.mean(tiempos_totales)), "energia": float(numpy.mean(energias)), "violacion": float(numpy.mean(excesos))})
    informe["pareto"] = compromiso

    # ------------------------------------------------ 3. Escalabilidad: problemas de distinto tamaño
    print("3/4 Escalabilidad...")
    escalabilidad = []
    for numero_de_tareas, numero_de_maquinas in [(50, 8), (100, 12), (200, 16), (400, 24)]:
        problema_de_otro_tamano = crear_problema(numero_de_tareas, numero_de_maquinas)
        fitness_voraz_de_este_tamano = describir_solucion(problema_de_otro_tamano, crear_solucion_voraz(problema_de_otro_tamano))[0]
        fila = {"n": numero_de_tareas, "m": numero_de_maquinas, "Voraz": fitness_voraz_de_este_tamano}
        for nombre in nombres_de_algoritmos:
            datos = ejecutar_varias_corridas(problema_de_otro_tamano, nombre, corridas_de_escalabilidad, tamano_de_poblacion, numero_de_iteraciones, semilla_inicial=3000)
            fila[nombre] = {"media": float(datos["fitness_final"].mean()), "std": float(datos["fitness_final"].std(ddof=1)), "tiempo": float(datos["tiempo"].mean())}
        escalabilidad.append(fila)
    informe["escalabilidad"] = escalabilidad

    # ------------------------------------------------ 4. Parametros adaptativos (una corrida del hibrido)
    generador_aleatorio = numpy.random.default_rng(1000)
    solucion, historial, registro = algoritmo_hibrido(problema, generador_aleatorio,  tamano_de_poblacion, numero_de_iteraciones)
    informe["adaptacion"] = {"pm_inicial": registro["mutacion"][0], "pm_final": registro["mutacion"][-1], "div_inicial": registro["diversidad"][0], "div_final": registro["diversidad"][-1]}

    # ------------------------------------------------ Graficas
    print("4/4 Dibujando graficas...")
    graficar_convergencia(resultados, fitness_del_voraz, numero_de_iteraciones, carpeta_de_salida)
    graficar_cajas(resultados, fitness_del_voraz, carpeta_de_salida)

    nombres_de_metodos = ["Round-Robin", "Voraz", "GA", "PSO", "GA-PSO"]
    tiempos_totales_por_metodo = [informe["base"]["Round-Robin"]["makespan"], informe["base"]["Voraz"]["makespan"]]
    energias_por_metodo = [informe["base"]["Round-Robin"]["energia"], informe["base"]["Voraz"]["energia"]]
    for nombre in nombres_de_algoritmos:
        tiempos_totales_por_metodo.append(informe["algoritmos"][nombre]["makespan_medio"])
        energias_por_metodo.append(informe["algoritmos"][nombre]["energia_media"])
    graficar_tiempo_y_energia(nombres_de_metodos, tiempos_totales_por_metodo, energias_por_metodo, carpeta_de_salida)

    posicion_de_la_mejor_corrida = resultados["GA-PSO"]["fitness_final"].argmin()
    mejor_solucion_hibrida = resultados["GA-PSO"]["soluciones"][posicion_de_la_mejor_corrida]
    graficar_asignacion(problema, solucion_round_robin, solucion_voraz, mejor_solucion_hibrida, carpeta_de_salida)
    graficar_parametros_adaptativos(registro, carpeta_de_salida)
    graficar_sensibilidad(sensibilidad, carpeta_de_salida)
    graficar_compromiso(compromiso, carpeta_de_salida)
    graficar_escalabilidad(escalabilidad, carpeta_de_salida)

    with open(os.path.join(carpeta_de_salida, "resultados.json"), "w") as archivo:
        json.dump(informe, archivo, indent=2, default=float)
    return informe

# =============================================================================
# INICIO DEL PROGRAMA
# =============================================================================
if __name__ == "__main__":
    modo_rapido = "--rapido" in sys.argv
    informe = ejecutar_experimento_completo(modo_rapido=modo_rapido)

    print("\n===== RESUMEN =====")
    print("Fitness del metodo voraz:", round(informe["base"]["Voraz"]["fitness"], 4))
    for nombre in ["GA", "PSO", "GA-PSO"]:
        fila = informe["algoritmos"][nombre]
        print(nombre, "- fitness medio:", round(fila["media"], 4),
              "| desviacion estandar:", round(fila["std"], 4),
              "| mejor:", round(fila["mejor"], 4))
    print("\nListo. Las graficas y los resultados estan en la carpeta 'resultados/'.")
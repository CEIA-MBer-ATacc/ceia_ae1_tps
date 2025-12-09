"""
GA para resolver laberintos usando DEAP 1.4.3 y pyamaze.
- Cromosoma: secuencia de movimientos (lista variable de ints 0..3 => U,D,L,R)
- Evaluación: simula el camino en el laberinto, penaliza movimientos inválidos,
  premia llegar a la meta y aproxima por distancia Manhattan.
- Soporta multiprocessing para acelerar la evaluación en poblaciones grandes.
- Requiere: deap==1.4.3, pyamaze
"""

import random
import math
from functools import partial
from collections import deque

# DEAP imports
from deap import base, creator, tools
# pyamaze
from pyamaze import maze, agent

# Optional multiprocessing
import multiprocessing

# ---------------------------
# Configuración del problema
# ---------------------------
ROWS = 100     # filas (alto)
COLS = 100     # columnas (ancho)

START = (1, 1)
GOAL = (ROWS, COLS)

# Longitudes mín/max de la secuencia (ajustar según problema).
MIN_LEN = (ROWS + COLS) // 2           # longitud mínima razonable
MAX_LEN = (ROWS * COLS) // 6           # reduce tamaño para manejar 100x100; ajustar si querés más libertad

# DEAP / GA hyperparámetros
POP_SIZE = 300
NGEN = 800
CXPB = 0.7
MUTPB = 0.35
TOURN_SIZE = 3
ELITE_SIZE = 4

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Mapeo movimientos: 0=U,1=D,2=L,3=R
MOVE_DELTAS = {
    0: (-1, 0),  # U
    1: (1, 0),   # D
    2: (0, -1),  # L
    3: (0, 1),   # R
}

# ---------------------------
# Generar laberinto con pyamaze
# ---------------------------
m = maze(ROWS, COLS)
# loopPercent=100 genera laberintos con bucles (más conectividad). Bajalo si querés arteros laberintos.
m.CreateMaze(loopPercent=100)
# agente para trazar luego
a = agent(m, footprints=True)


# ---------------------------
# Utilidades para simular
# ---------------------------
def can_move(cell, move):
    """Devuelve True si desde `cell` se puede mover en la dirección `move` según m.maze_map"""
    r, c = cell
    d = MOVE_DELTAS[move]
    nr, nc = r + d[0], c + d[1]
    # verificar límites
    if not (1 <= nr <= ROWS and 1 <= nc <= COLS):
        return False
    # Determinar la clave de pared según delta
    if d == (1, 0):
        return m.maze_map[cell]["S"]
    if d == (-1, 0):
        return m.maze_map[cell]["N"]
    if d == (0, 1):
        return m.maze_map[cell]["E"]
    if d == (0, -1):
        return m.maze_map[cell]["W"]
    return False


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def simulate_individual(individual, stop_on_goal=True):
    """
    Simula la ejecución del individuo (secuencia de movimientos).
    Retorna: reached(bool), steps_taken (int), invalid_moves (int), final_pos (tuple)
    """
    pos = START
    steps = 0
    invalid = 0
    for move in individual:
        steps += 1
        if can_move(pos, move):
            d = MOVE_DELTAS[move]
            pos = (pos[0] + d[0], pos[1] + d[1])
        else:
            invalid += 1
            # quedamos en la misma celda
        if stop_on_goal and pos == GOAL:
            return True, steps, invalid, pos
    return (pos == GOAL), steps, invalid, pos


# ---------------------------
# DEAP: definición del individuo y toolbox
# ---------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximizamos
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generador de movimientos (0..3)
toolbox.register("attr_move", random.randint, 0, 3)

# Generador de longitud variable
def init_individual(min_len=MIN_LEN, max_len=MAX_LEN):
    L = random.randint(min_len, max_len)
    return creator.Individual([toolbox.attr_move() for _ in range(L)])

toolbox.register("individual", init_individual, MIN_LEN, MAX_LEN)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness: simula y calcula una función que premia llegar y acercarse
def evaluate_fitness(individual, weight_reach=1e6, weight_step=1.0, weight_invalid=50.0, weight_dist=10.0):
    reached, steps, invalid, final_pos = simulate_individual(individual, stop_on_goal=True)
    dist = manhattan(final_pos, GOAL)

    # Componentes:
    score = 0.0
    if reached:
        # recompensa alta por alcanzar la meta: entre más rápido mejor
        score += weight_reach + (max(0, (MAX_LEN - steps)) * weight_step)
    else:
        # no llegó: se premia por acercarse y por no tener muchos invalid moves
        # usamos inversa de distancia para que más cerca => mayor score
        score += (1.0 / (dist + 1.0)) * (weight_reach * 0.001)  # escala pequeña
        score += ( (len(individual) - invalid) * 0.5 )  # favorecer rutas con muchos movimientos válidos

    # penalizar invalid moves
    score -= invalid * weight_invalid

    # penalizar longitud excesiva (para favorecer soluciones más cortas)
    score -= len(individual) * 0.01

    return (score,)

toolbox.register("evaluate", evaluate_fitness)

# Operadores genéticos para individuos de longitud variable
def cx_one_point_var(ind1, ind2):
    """Cruce: intercambia colas a partir de puntos aleatorios (soporta distinta longitud)."""
    if len(ind1) == 0 or len(ind2) == 0:
        return ind1, ind2
    cx1 = random.randint(1, len(ind1) - 1) if len(ind1) > 1 else 0
    cx2 = random.randint(1, len(ind2) - 1) if len(ind2) > 1 else 0
    new1 = ind1[:cx1] + ind2[cx2:]
    new2 = ind2[:cx2] + ind1[cx1:]
    # limitar longitud a MAX_LEN
    del ind1[:]
    del ind2[:]
    ind1.extend(new1[:MAX_LEN])
    ind2.extend(new2[:MAX_LEN])
    return ind1, ind2

def mut_individual(individual, indpb=0.05):
    """
    Mutación variable:
    - con prob replace: cambia un gen al azar
    - con prob insert: inserta un gen aleatorio
    - con prob delete: borra un gen (siempre que quede >= MIN_LEN)
    indpb: probabilidad por sitio de aplicar reemplazo
    """
    # Reemplazar algunos genes
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = toolbox.attr_move()

    # Inserción
    if random.random() < 0.15 and len(individual) < MAX_LEN:
        pos = random.randint(0, len(individual))
        individual.insert(pos, toolbox.attr_move())

    # Borrado
    if random.random() < 0.10 and len(individual) > MIN_LEN:
        pos = random.randint(0, len(individual) - 1)
        individual.pop(pos)

    # Enforce bounds
    if len(individual) > MAX_LEN:
        del individual[MAX_LEN:]
    if len(individual) < 1:
        individual.extend([toolbox.attr_move() for _ in range(MIN_LEN)])
    return (individual,)

toolbox.register("mate", cx_one_point_var)
toolbox.register("mutate", mut_individual, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

# opcional: usar multiprocessing para acelerar la evaluación
USE_MP = True
pool = None
if USE_MP:
    try:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    except Exception as e:
        print("No se pudo iniciar multiprocessing pool:", e)
        USE_MP = False

# ---------------------------
# Bucle principal del GA
# ---------------------------
def run_ga(verbose=True):
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(ELITE_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda vs: sum([v[0] for v in vs]) / len(vs))
    stats.register("max", lambda vs: max(v[0] for v in vs))
    stats.register("min", lambda vs: min(v[0] for v in vs))

    # Evaluar población inicial
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if verbose:
        print("Inicio GA: POP={}, NGEN={}, LENmin={}, LENmax={}".format(POP_SIZE, NGEN, MIN_LEN, MAX_LEN))

    best_reached = False
    best_gen = None

    for gen in range(1, NGEN + 1):
        # Elitismo: conservar los mejores
        elites = tools.selBest(pop, ELITE_SIZE)

        # Selección y emparejamiento
        offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
        offspring = list(map(toolbox.clone, offspring))

        # Cruce
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < CXPB:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutación
        for i in range(len(offspring)):
            if random.random() < MUTPB:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Recomponer población
        pop = elites + offspring

        # Evaluar individuos sin fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Estadísticas, HOF
        hof.update(pop)
        record = stats.compile(pop)
        if verbose and gen % 10 == 0:
            print("Gen {:4d} | max {:.2f} | avg {:.2f} | min {:.2f}".format(gen, record["max"], record["avg"], record["min"]))

        # Chequear si mejor llegó a la meta
        best = tools.selBest(pop, 1)[0]
        reached, steps, invalid, final_pos = simulate_individual(best, stop_on_goal=True)
        if reached:
            best_reached = True
            best_gen = (gen, best, steps, invalid)
            print("¡Solución encontrada en gen {}! steps={} invalid={}".format(gen, steps, invalid))
            break

    if not best_reached:
        print("No se encontró solución dentro de las generaciones. Mejor individuo:")
        best = tools.selBest(pop, 1)[0]
        reached, steps, invalid, final_pos = simulate_individual(best, stop_on_goal=True)
        print(" Mejor fitness {:.2f} | reached={} | steps={} | invalid={} | final_pos={}".format(best.fitness.values[0], reached, steps, invalid, final_pos))

    # Mostrar y trazar la mejor ruta si llegó
    if best_reached:
        gen, best_ind, steps, invalid = best_gen
        print("Solución final (gen {}): len={} steps={} invalid={}".format(gen, len(best_ind), steps, invalid))
        # Simular trayecto completo y construir path (lista de coordenadas)
        pos = START
        path = [pos]
        for mv in best_ind:
            if can_move(pos, mv):
                d = MOVE_DELTAS[mv]
                pos = (pos[0] + d[0], pos[1] + d[1])
                path.append(pos)
            if pos == GOAL:
                break
        # Convertir path a formato que pyamaze tracePath usa (lista de tuples)
        # pyamaze espera coords (row, col)
        if path[-1] == GOAL:
            # Trace path using pyamaze
            # build a dict {agent: path_in_pyamaze_format}
            py_path = list(path)  # ya en formato (r,c)
            m.tracePath({a: py_path}, showMarked=True)
            m.run()

    if USE_MP and pool:
        pool.close()
        pool.join()

    return hof

# ---------------------------
# Ejecutar
# ---------------------------
if __name__ == "__main__":
    hof = run_ga(verbose=True)
    print("HallOfFame (top individuals):")
    for i, ind in enumerate(hof):
        print(i, "len", len(ind), "fitness", ind.fitness.values[0])

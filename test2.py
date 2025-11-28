import random
import numpy as np

import pygame
import sys
import time

# --------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------

CELL_SIZE = 20          # tamaño de cada celda en píxeles
WALL_COLOR = (40, 40, 40)
FREE_COLOR = (220, 220, 220)
PATH_COLOR = (255, 0, 0)
AGENT_COLOR = (0, 0, 255)
FPS = 10                # velocidad del movimiento (frames por segundo)

MOVES = {
    1: (1, 0),   # right
    2: (0, 1),   # up
    3: (-1, 0),  # left
    4: (0, -1)   # down
}

class GeneticMazeSolver:
    def __init__(self, maze, start, goal,
                 pop_size=100, max_gen=20, max_len=20):
        self.maze = maze
        self.start = start
        self.goal = goal

        self.pop_size = pop_size
        self.max_gen = max_gen
        self.max_len = max_len

        # Coeficientes del paper:
        self.k1 = 1.0
        self.k2 = 1.0
        self.k3 = 400.0
        self.k4 = 200.0

    # =====================
    # DECODIFICACIÓN DEL CROMOSOMA
    # =====================
    def simulate(self, genes):
        x, y = self.start
        path = [(x, y)]
        crash = 0
        movement_changes = 0
        pen = 0
        for i, g in enumerate(genes):
            dx, dy = MOVES[g]
            nx, ny = x + dx, y + dy

            # condición pared o fuera
            if self.collides(nx, ny):
                crash += 1
                movement_changes += 1
                pen += 1
                continue  # sigue usando mismo gen

            # mover
            x, y = nx, ny
            path.append((x, y))

            # punto de decisión
            if self.is_decision(x, y):
                movement_changes += 1

            # objetivo
            if (x, y) == self.goal:
                break

        return path, crash, movement_changes, pen

    # =====================
    # COLISIÓN CON PARED
    # =====================
    def collides(self, x, y):
        if x < 0 or y < 0 or y >= len(self.maze) or x >= len(self.maze[0]):
            return True
        try:
            return self.maze[y][x] == 1
        except:
            return True

    # =====================
    # PUNTO DE DECISIÓN
    # =====================
    def is_decision(self, x, y):
        free = 0
        for m in MOVES.values():
            nx, ny = x + m[0], y + m[1]
            if not self.collides(nx, ny):
                free += 1
        return free > 2

    # =====================
    # FUNCIÓN FITNESS
    # =====================
    def fitness(self, individual):
        path, crash, m_changes, pen = self.simulate(individual)
        last = path[-1]

        dist = np.linalg.norm(np.array(last) - np.array(self.goal))
        steps = len(path)

        return dist + self.k1*m_changes + self.k2*steps + self.k3*crash + self.k4*pen

    # =====================
    # SELECCIÓN
    # =====================
    def select(self, population, scores):
        # Ranking selection
        sorted_pop = [x for _, x in sorted(zip(scores, population))]
        return sorted_pop[:len(sorted_pop)//2]

    # =====================
    # CROSSOVER (2 puntos)
    # =====================
    def crossover(self, p1, p2):
        n = min(len(p1), len(p2))

        # primer corte principal
        PC = random.randint(1, n-2)

        # subcortes alrededor del PC
        Pcsub1 = random.randint(1, PC)
        Pcsub2 = random.randint(PC, n-1)

        c1 = p1[:Pcsub1] + p2[Pcsub1:Pcsub2] + p1[Pcsub2:]
        c2 = p2[:Pcsub1] + p1[Pcsub1:Pcsub2] + p2[Pcsub2:]

        return c1, c2

    # =====================
    # MUTACIÓN
    # =====================
    def mutate(self, individual):
        if random.random() < 0.8: return individual
        idx = random.randint(0, len(individual)-1)
        individual[idx] = random.randint(1, 4)
        return individual

    # =====================
    # LOOP GA
    # =====================
    def solve(self):
        population = [self.random_individual() for _ in range(self.pop_size)]

        for gen in range(self.max_gen):
            scores = [self.fitness(ind) for ind in population]
            best = population[np.argmin(scores)]
            print(f"Gen {gen} score: {min(scores):.3f} best path len={len(best)}")

            # Encontró meta
            if self.simulate(best)[0][-1] == self.goal:
                print("💥 Goal reached!")
                return best

            selected = self.select(population, scores)
            children = []

            while len(children) < self.pop_size:
                p1, p2 = random.sample(selected, 2)
                c1, c2 = self.crossover(p1, p2)
                children.append(self.mutate(c1))
                children.append(self.mutate(c2))

            population = children[:self.pop_size]

        return best

    def random_individual(self):
        return [random.randint(1, 4) for _ in range(self.max_len)]



# --------------------------------------------
# FUNCIÓN PRINCIPAL
# --------------------------------------------

def draw_maze(screen, maze):
    """Dibuja el laberinto completo"""
    rows = len(maze)
    cols = len(maze[0])

    for y in range(rows):
        for x in range(cols):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if maze[y][x] == 1:  # pared
                pygame.draw.rect(screen, WALL_COLOR, rect)
            else:               # espacio libre
                pygame.draw.rect(screen, FREE_COLOR, rect)

            # bordes opcionales
            pygame.draw.rect(screen, (180, 180, 180), rect, 1)


def animate_path(screen, maze, path):
    """Dibuja el camino paso a paso con un agente moviéndose"""
    draw_maze(screen, maze)

    # Dibuja líneas del camino recorrido
    for i in range(1, len(path)):
        x1 = path[i - 1][1] * CELL_SIZE + CELL_SIZE // 2
        y1 = path[i - 1][0] * CELL_SIZE + CELL_SIZE // 2
        x2 = path[i][1] * CELL_SIZE + CELL_SIZE // 2
        y2 = path[i][0] * CELL_SIZE + CELL_SIZE // 2

        pygame.draw.line(screen, PATH_COLOR, (x1, y1), (x2, y2), 3)

        # Dibujar agente
        ax = path[i][1] * CELL_SIZE + CELL_SIZE // 2
        ay = path[i][0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, AGENT_COLOR, (ax, ay), CELL_SIZE // 3)

        pygame.display.update()
        clock.tick(FPS)

    # Dejar el resultado final
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()




if __name__ == "__main__":
    maze = [
        [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
        [1,1,0,1,0,1,1,1,1,1,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
        [0,1,1,1,1,1,1,1,0,1,1,1,0,1,0],
        [0,0,0,0,0,0,0,1,0,0,0,1,0,1,0],
        [1,1,1,1,1,1,0,1,1,1,0,1,0,1,0],
        [0,0,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
        [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
        [0,1,0,1,1,1,0,1,1,1,0,1,1,1,0],
        [0,1,0,0,0,0,0,0,0,1,0,0,0,1,0],
        [0,1,1,1,1,1,1,1,0,1,1,1,0,1,0],
        [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],
        [1,1,1,1,1,1,0,1,1,1,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]


    solver = GeneticMazeSolver(
        maze,
        start = (0,0),
        goal = (14,14),
        pop_size = 300,
        max_gen = 150,
        max_len = 180
    )

    solution = solver.solve()
    print("\nSOLUTION:", solution)
    path,_ ,_, _= solver.simulate(solution)
    print("PATH:", path)


    pygame.init()
    clock = pygame.time.Clock()

    height = len(maze) * CELL_SIZE
    width  = len(maze[0]) * CELL_SIZE

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Laberinto con recorrido - Pygame")

    animate_path(screen, maze, path)
import random
import numpy as np

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
        self.k3 = 40.0
        self.k4 = 200.0

    # =====================
    # DECODIFICACIÓN DEL CROMOSOMA
    # =====================
    def simulate(self, genes):
        x, y = self.start
        path = [(x, y)]
        crash = 0
        movement_changes = 0

        for i, g in enumerate(genes):
            dx, dy = MOVES[g]
            nx, ny = x + dx, y + dy

            # condición pared o fuera
            if self.collides(nx, ny):
                crash += 1
                movement_changes += 1
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

        return path, crash, movement_changes

    # =====================
    # COLISIÓN CON PARED
    # =====================
    def collides(self, x, y):
        if x < 0 or y < 0:
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
        path, crash, m_changes = self.simulate(individual)
        last = path[-1]

        dist = np.linalg.norm(np.array(last) - np.array(self.goal))
        steps = len(path)

        return dist + self.k1*m_changes + self.k2*steps + self.k3*crash

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
        max_gen = 100,
        max_len = 150
    )

    solution = solver.solve()
    print("\nSOLUTION:", solution)
    path,_ ,_= solver.simulate(solution)
    print("PATH:", path)


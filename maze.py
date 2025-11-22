# maze_evo.py
import pygame
import random
import numpy as np
import math
import os
from collections import deque

# ---------------------------
# Maze generator (DFS)
# ---------------------------
def generate_maze(width=21, height=21, seed=None):
    assert width % 2 == 1 and height % 2 == 1, "Width/Height must be odd"
    if seed is not None:
        random.seed(seed)
    maze = [["#" for _ in range(width)] for _ in range(height)]

    def carve(x, y):
        dirs = [(2,0),(-2,0),(0,2),(0,-2)]
        random.shuffle(dirs)
        for dx,dy in dirs:
            nx, ny = x+dx, y+dy
            if 1 <= nx < width-1 and 1 <= ny < height-1 and maze[ny][nx] == "#":
                maze[ny - dy//2][nx - dx//2] = " "
                maze[ny][nx] = " "
                carve(nx, ny)

    maze[1][1] = " "
    carve(1,1)
    return maze

# ---------------------------
# Environment wrapper
# ---------------------------
class MazeEnv:
    def __init__(self, maze, tile_size=20, render=False, max_steps=500):
        self.maze = maze
        self.h = len(maze)
        self.w = len(maze[0])
        self.tile = tile_size
        self.render_mode = render
        self.max_steps = max_steps

        # start at (1,1) and goal at bottom-right open cell
        self.start = (1,1)
        # find goal: nearest open to bottom-right
        for y in range(self.h-2,0,-1):
            for x in range(self.w-2,0,-1):
                if maze[y][x] == " ":
                    self.goal = (x,y)
                    break
            else:
                continue
            break

        # Pygame init
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.w*self.tile, self.h*self.tile))
            pygame.display.set_caption("Maze Evolution")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.pos = list(self.start)
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        # actions: 0 up,1 down,2 left,3 right
        dx, dy = 0,0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1

        nx = self.pos[0] + dx
        ny = self.pos[1] + dy
        if 0 <= nx < self.w and 0 <= ny < self.h and self.maze[ny][nx] == " ":
            self.pos = [nx, ny]  # move
        self.steps += 1

        done = (tuple(self.pos) == self.goal) or (self.steps >= self.max_steps)
        # reward = negative distance to goal (we'll use fitness externally)
        obs = self._get_obs()
        return obs, 0.0, done, {}

    def _get_obs(self):
        # sensors: distances to wall in 4 cardinal directions (normalized)
        max_dist = max(self.w, self.h)
        dists = []
        # up
        dy = 0
        for s in range(1, max_dist):
            y = self.pos[1] - s
            if y < 0 or self.maze[y][self.pos[0]] == "#":
                dists.append(s)
                break
        # down
        for s in range(1, max_dist):
            y = self.pos[1] + s
            if y >= self.h or self.maze[y][self.pos[0]] == "#":
                dists.append(s)
                break
        # left
        for s in range(1, max_dist):
            x = self.pos[0] - s
            if x < 0 or self.maze[self.pos[1]][x] == "#":
                dists.append(s)
                break
        # right
        for s in range(1, max_dist):
            x = self.pos[0] + s
            if x >= self.w or self.maze[self.pos[1]][x] == "#":
                dists.append(s)
                break

        # vector to goal (dx, dy), normalized to [-1,1]
        dx = (self.goal[0] - self.pos[0]) / self.w
        dy = (self.goal[1] - self.pos[1]) / self.h

        # normalize dists
        dists_norm = [d / max_dist for d in dists]
        obs = np.array(dists_norm + [dx, dy], dtype=np.float32)
        return obs

    def render(self):
        if not self.render_mode:
            return

        # si no existe clock, inicializar pygame ahora
        if not hasattr(self, "clock"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.w*self.tile, self.h*self.tile))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                os._exit(0)

        self.screen.fill((0,0,0))
        for y,row in enumerate(self.maze):
            for x,ch in enumerate(row):
                if ch == "#":
                    pygame.draw.rect(self.screen, (0,0,0), (x*self.tile,y*self.tile,self.tile,self.tile))
                    pygame.draw.rect(self.screen, (100,100,100), (x*self.tile,y*self.tile,self.tile,self.tile),1)
                else:
                    pygame.draw.rect(self.screen, (255,255,255), (x*self.tile,y*self.tile,self.tile,self.tile))

        # agent
        ax,ay = self.pos
        pygame.draw.circle(self.screen, (0,0,255),
                        (ax*self.tile + self.tile//2, ay*self.tile + self.tile//2),
                        self.tile//3)

        # goal
        gx,gy = self.goal
        pygame.draw.rect(self.screen, (0,255,0),
                        (gx*self.tile, gy*self.tile, self.tile, self.tile))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode:
            pygame.quit()

# ---------------------------
# Simple feed-forward NN (weights from genome)
# ---------------------------
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, genome=None):
        self.in_sz = input_size
        self.hid = hidden_size
        self.out_sz = output_size
        # genome layout: W1 (in*hid), b1 (hid), W2 (hid*out), b2 (out)
        self.genome_len = input_size*hidden_size + hidden_size + hidden_size*output_size + output_size
        if genome is None:
            # initialize small weights
            self.genome = np.random.randn(self.genome_len) * 0.5
        else:
            self.genome = genome.copy()

        # map genome to weights
        self._unpack()

    def _unpack(self):
        idx = 0
        s = self.in_sz * self.hid
        self.W1 = self.genome[idx:idx+s].reshape((self.in_sz, self.hid)); idx += s
        s = self.hid
        self.b1 = self.genome[idx:idx+s]; idx += s
        s = self.hid * self.out_sz
        self.W2 = self.genome[idx:idx+s].reshape((self.hid, self.out_sz)); idx += s
        s = self.out_sz
        self.b2 = self.genome[idx:idx+s]; idx += s

    def forward(self, x):
        h = np.tanh(x.dot(self.W1) + self.b1)
        out = (h.dot(self.W2) + self.b2)
        return out

    def act(self, obs):
        logits = self.forward(obs)
        # choose action with highest logit
        action = int(np.argmax(logits))
        return action

# ---------------------------
# Genetic Algorithm
# ---------------------------
def evaluate_genome(genome, env, render=False, episodes=1):
    nn = SimpleNN(input_size=6, hidden_size=12, output_size=4, genome=genome)
    total_score = 0.0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        for _ in range(env.max_steps):
            if render:
                env.render()
            action = nn.act(obs)
            obs, _, done, _ = env.step(action)
            steps += 1
            if done:
                break
        # fitness: inverse distance to goal + big bonus if reached
        pos = env.pos
        goal = env.goal
        dist = math.hypot(goal[0]-pos[0], goal[1]-pos[1])
        # normalize by diag
        diag = math.hypot(env.w, env.h)
        base = 1.0 - (dist / diag)  # closer -> higher (0..1)
        bonus = 1.0 if tuple(pos) == goal else 0.0
        # also reward speed: if reached, add (remaining_steps/max_steps)
        speed_bonus = 0.0
        if bonus > 0:
            speed_bonus = (env.max_steps - steps) / env.max_steps
        score = base + bonus*2.0 + speed_bonus
        total_score += score
    return total_score / episodes

def tournament_selection(pop, fitnesses, k=3):
    idxs = np.arange(len(pop))
    chosen = []
    for _ in range(len(pop)):
        aspirants = np.random.choice(idxs, k, replace=False)
        winner = aspirants[np.argmax(fitnesses[aspirants])]
        chosen.append(pop[winner].copy())
    return chosen

def crossover(a, b):
    # uniform crossover
    mask = np.random.rand(len(a)) < 0.5
    child = a.copy()
    child[mask] = b[mask]
    return child

def mutate(genome, mutation_rate=0.1, mutation_power=0.3):
    # gaussian mutation per gene
    mask = np.random.rand(len(genome)) < mutation_rate
    genome[mask] += np.random.randn(np.sum(mask)) * mutation_power
    return genome

# ---------------------------
# Main evolution loop
# ---------------------------
def evolve(env, generations=60, pop_size=80, render_eval=False):
    # create initial population
    sample_nn = SimpleNN(6,12,4)
    glen = sample_nn.genome_len
    pop = [np.random.randn(glen) * 0.5 for _ in range(pop_size)]
    best = None
    best_fitness = -1e9

    for gen in range(1, generations+1):
        fitnesses = np.zeros(pop_size)
        for i,genome in enumerate(pop):
            fitnesses[i] = evaluate_genome(genome, env, render=False, episodes=1)
        # logging
        avg = np.mean(fitnesses)
        mx = np.max(fitnesses)
        mi = np.min(fitnesses)
        best_idx = int(np.argmax(fitnesses))
        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best = pop[best_idx].copy()
        print(f"Gen {gen}/{generations} | avg {avg:.3f} | max {mx:.3f} | min {mi:.3f} | best_so_far {best_fitness:.3f}")

        # selection
        selected = tournament_selection(pop, fitnesses, k=3)
        # create offspring
        newpop = []
        # elitism: keep top 2
        elite_idxs = fitnesses.argsort()[-2:][::-1]
        newpop.append(pop[elite_idxs[0]].copy())
        newpop.append(pop[elite_idxs[1]].copy())

        while len(newpop) < pop_size:
            a = random.choice(selected)
            b = random.choice(selected)
            child = crossover(a, b)
            child = mutate(child, mutation_rate=0.08, mutation_power=0.2)
            newpop.append(child)
        pop = newpop

    return best, best_fitness

# ---------------------------
# Save / Load helpers
# ---------------------------
def save_genome(genome, filename="best_genome.npy"):
    np.save(filename, genome)
    print("Saved genome to", filename)

def load_genome(filename="best_genome.npy"):
    return np.load(filename)

# ---------------------------
# Run: build maze, evolve, and demo best
# ---------------------------
if __name__ == "__main__":
    # Hyperparams
    MAZE_W = 31
    MAZE_H = 21
    SEED = 42
    GENERATIONS = 200
    POP_SIZE = 300
    MAX_STEPS = 1000

    maze = generate_maze(MAZE_W, MAZE_H, seed=SEED)
    env = MazeEnv(maze, tile_size=16, render=False, max_steps=MAX_STEPS)

    print("Maze size:", MAZE_W, MAZE_H, "Start:", env.start, "Goal:", env.goal)
    best_genome, best_fit = evolve(env, generations=GENERATIONS, pop_size=POP_SIZE, render_eval=False)
    print("Best fitness:", best_fit)

    save_genome(best_genome, "best_genome.npy")

    # Demo with rendering
    print("Demoing best genome. Close the window to exit.")
    env.render_mode = True
    env.screen = pygame.display.set_mode((env.w*env.tile, env.h*env.tile))
    demo_nn = SimpleNN(6,12,4, genome=best_genome)
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = demo_nn.act(obs)
        obs, _, done, _ = env.step(action)
    env.render()
    pygame.time.wait(1000)
    env.close()

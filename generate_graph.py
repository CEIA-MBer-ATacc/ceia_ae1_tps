import matplotlib.pyplot as plt
from maze_solver import GeneticMazeSolver

# Configuración del laberinto y del solver
# Se utiliza el mismo laberinto y configuración que en el script principal
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
    start=(0, 0),
    goal=(14, 14),
    pop_size=500,
    max_gen=1000,
    min_len=30,
    max_len=180
)

# Ejecutar el solver y obtener el historial de fitness
print("Ejecutando el Algoritmo Genético para generar el gráfico...")
solution, fitness_history = solver.solve()
print("Ejecución finalizada.")

# Generar el gráfico de convergencia
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, label='Mejor Fitness por Generación')
plt.title('Gráfico de Convergencia del Algoritmo Genético')
plt.xlabel('Generación')
plt.ylabel('Fitness Score (Menor es Mejor)')
plt.grid(True)
plt.legend()

# Guardar el gráfico en un archivo
file_path = 'convergence.png'
plt.savefig(file_path)

print(f"Gráfico de convergencia guardado en: {file_path}")

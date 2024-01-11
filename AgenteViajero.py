
import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 20           # Tamaño del ADN
CROSS_RATE = 0.1        # Tasa de cruce
MUTATE_RATE = 0.02      # Tasa de mutación
POP_SIZE = 500          # Tamaño de la población
N_GENERATIONS = 500     # Numero de Generaciones

class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)]) #  Matriz que contiene la población actual

    def translateDNA(self, DNA, city_position):     # Ordena las coordenadas de las ciudades
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y): # Calcula la aptitud de cada individuo
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness): # Selecciona individuos basándose en su aptitud.
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop): # Aplica el operador de cruce
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # selecciona otro individuo del pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(bool)      # elegir los puntos de cruce
            keep_city = parent[~cross_points]                                       # encontrar el número de la ciudad
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child): # Aplica el operador de mutación
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness): # Realiza la evolución mediante la selección, cruce y mutación
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # para cada padre
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class PlotCities(object):
    def __init__(self, n_cities):
        self.city_names = ['Lima', 'Arequipa', 'Cusco', 'Trujillo', 'Chiclayo', 'Piura', 'Iquitos', 'Huancayo', 'Tacna', 'Cajamarca',
                            'Chimbote', 'Pucallpa', 'Huaraz', 'Ica', 'Juliaca', 'Tarapoto', 'Ayacucho', 'Puno', 'Cerro de Pasco',
                            'Chincha Alta', 'Huánuco', 'Tumbes', 'Moquegua', 'Abancay', 'Paita', 'Catacaos', 'Sullana', 'Jaén', 'Tarma']
        self.city_position = np.random.rand(n_cities, 2)
        self.city_dict = dict(zip(self.city_names, self.city_position))
        plt.ion()

    def plot_cities(self): # grafica los nombres de las ciudades
        for city_name, (x, y) in self.city_dict.items():
            plt.text(x, y, city_name, fontsize=8, ha='center', va='center')

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        self.plot_cities() 
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Distancia Total=%.2f km" % total_d, fontdict={'size': 20, 'color': 'purple'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

# EJECUCION DEL PROGRAMA
if __name__ == '__main__':
    
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

    plotGA = PlotCities(N_CITIES)

    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, plotGA.city_position)
        fitness, total_distance = ga.get_fitness(lx, ly)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)

        plotGA.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

    plt.ioff()
    plt.show()
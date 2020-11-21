import copy
import numpy as np

from utils import Individual, accuracy_score


class BaseOptimizer:

    def __init__(self, params):
        self.params = [param for param in params]

    def update_params(self, grads=None):
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

    def iteration_ends(self, time_step):
        pass

    def trigger_stopping(self):
        return True


class SGDOptimizer(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.1, lr_schedule='constant',
                 momentum=0.9, nesterov=True, power_t=0.5):
        super().__init__(params)
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.velocities = [np.zeros_like(param) for param in params]

    def iteration_ends(self, time_step):
        if self.lr_schedule == 'invscaling':
            self.learning_rate = (float(self.learning_rate_init) /
                                  (time_step + 1) ** self.power_t)

    def trigger_stopping(self):
        if self.lr_schedule != 'adaptive':
            return True
        if self.learning_rate <= 1e-6:
            return True
        self.learning_rate /= 5.
        return False

    def _get_updates(self, grads):
        updates = [self.momentum * velocity - self.learning_rate * grad
                   for velocity, grad in zip(self.velocities, grads)]
        self.velocities = updates

        if self.nesterov:
            updates = [self.momentum * velocity - self.learning_rate * grad
                       for velocity, grad in zip(self.velocities, grads)]

        return updates


class AdamOptimizer(BaseOptimizer):

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        super().__init__(params)
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates


class GAOptimizer(BaseOptimizer):

    def __init__(self, model, X, y, params, pop_size=50, crossover_rate=0.8, mutation_rate=0.05):
        super().__init__(params)
        self.model = copy.deepcopy(model)
        self.X = X
        self.y = y
        self.chromosome_len = len(params)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.pop = []
        self.mating_pop = []
        for _ in range(self.pop_size):
            chromosome = []
            for param in params:
                chromosome.append(np.random.uniform(-1.0, 1.0, param.shape))
            fitness = self.get_fitness(chromosome)
            self.pop.append(Individual(chromosome, fitness))
            self.mating_pop.append(Individual(chromosome, fitness))
        self.update_params()

    def update_params(self, grads=None):
        updates = self._get_updates()
        for param, update in zip(self.params, updates):
            for i, j in zip(np.nditer(param, op_flags=['readwrite']), np.nditer(update, op_flags=['readwrite'])):
                i[...] = j[...]

    def _get_updates(self):
        self.select()
        self.crossover()
        self.mutate()
        for i in range(self.pop_size):
            self.pop[i].set_fitness(self.get_fitness(self.pop[i].chromosome()))
        self.pop[self.pop.index(min(self.pop, key=lambda x: x.fitness()))] = Individual(self.params,
                                                                                        self.get_fitness(self.params))
        return max(self.pop, key=lambda x: x.fitness()).chromosome()

    def get_fitness(self, chromosome):
        self.model.set_params(chromosome[: int(self.chromosome_len / 2)], chromosome[int(self.chromosome_len / 2):])
        fitness = accuracy_score(self.y, self.model.predict(self.X))
        return fitness

    def select(self):
        probs = []
        total_fitness = 0.0
        for individual in self.pop:
            total_fitness += individual.fitness()
        for individual in self.pop:
            probs.append(individual.fitness() / total_fitness)
        indices = np.random.choice(self.pop_size, self.pop_size - 1, p=probs)
        for i in range(self.pop_size - 1):
            self.mating_pop[i] = Individual(self.pop[indices[i]].chromosome(), self.pop[indices[i]].fitness())
        max_individual = max(self.pop, key=lambda x: x.fitness())
        self.mating_pop[-1] = Individual(max_individual.chromosome(), max_individual.fitness())

    def crossover(self):
        np.random.shuffle(self.mating_pop)
        for i in range(0, self.pop_size, 2):
            if np.random.rand() < self.crossover_rate:
                for j in range(self.chromosome_len):
                    chromosome1 = self.mating_pop[i].chromosome()[j]
                    chromosome2 = self.mating_pop[i + 1].chromosome()[j]
                    chromosome1_flat = chromosome1.flatten()
                    chromosome2_flat = chromosome2.flatten()
                    indices = np.random.choice(len(chromosome1_flat), size=2)
                    begin, end = np.sort(indices)
                    tmp = np.copy(chromosome1_flat[begin: end])
                    chromosome1_flat[begin: end] = chromosome2_flat[begin: end]
                    chromosome2_flat[begin: end] = tmp
                    self.mating_pop[i].chromosome()[j] = chromosome1_flat.reshape(chromosome1.shape)
                    self.mating_pop[i + 1].chromosome()[j] = chromosome2_flat.reshape(chromosome2.shape)
        self.pop = copy.deepcopy(self.mating_pop)

    def mutate(self):
        for i in range(self.pop_size):
            for j in range(self.chromosome_len):
                for val in np.nditer(self.pop[i].chromosome()[j], op_flags=['readwrite']):
                    if np.random.rand() < self.mutation_rate:
                        val[...] += np.random.normal(0.0, 0.1)

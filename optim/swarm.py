"""Swarm management
"""

import numpy as np

class Individual:
    def __init__(self, code, evaluation_f):
        self.code = code
        self.objective=evaluation_f
    def evaluate(self):
        return self.objective(self.code)

class Swarm:
    def __init__(self, initial_population=[],reproduction=None, population_control=None):
        self.population = initial_population
        self.epoch = 0
        self.reproduction = reproduction
        self.population_control = population_control
    def start(self):
        self.epoch = 0


        while self.epoch < 1000:
            # evaluation of the population
            evaluations = [a.evaluate() for a in self.population]
            print(self.epoch, np.min(evaluations), self.population[np.argmin(evaluations)].code, len(self.population))
            # apply reproduction strategy that depends on population evaluations
            new_population = self.reproduction(self.population, evaluations)
            # apply population control strategy
            self.population = self.population_control(new_population, evaluations)
            self.epoch += 1
            


if __name__=="__main__":
    from optim.test_function import Rastrigin
    max_population = 100
    def F(x):
        return np.sum(x**2)
    x0 = np.random.random_sample(2)*100.

    # first individual
    ind_0 = Individual(x0, Rastrigin(n=2))

    # reproduction
    def rep(population, evaluations):
        best_ind = population[np.argmin(evaluations)]
        new_ind = Individual(best_ind.code + 2.*(np.random.random_sample(size=best_ind.code.shape)-.5),
                best_ind.objective)
        return population+[new_ind]
    # population control
    def popc(population, evaluations):
        if len(population)>10:
            evs=[p.evaluate() for p in population]
            i = np.argsort(evs)[:10]
            return np.array(population,dtype=object)[i].tolist()
        return population


    swarm = Swarm(reproduction=rep, population_control=popc,initial_population=[ind_0])
    swarm.start()


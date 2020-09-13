import numpy as np
from pystrand_pkg.Genotype import Genotype
from pystrand_pkg.Population import Population

class Optimizer(object):
    """description of class"""
    _population = None
    _max_iterations = 0
    _target_genotype = np.array([])
    _mutation_probability = 0.0
    _crossover_probability = 0.0
    _selection_method = None

    def __init__(self, 
                 target_genotype, 
                 max_iterations,
                 mutation_prob = 0.001,
                 crossover_prob = 0.0,
                 selection_method = 'top',
                 population = None,
                 outfile='', 
                 *args, 
                 **kwargs):
        
        self._target_genotype = target_genotype
        self._mutation_probability = mutation_prob
        self._crossover_probability = crossover_prob
        self._selection_method = selection_method

        if population is not None:
            self._population = population
        else:
            self._population = Population(
                target_genotype.shape()**2, 
                target_genotype.shape())

        self._max_iterations = max_iterations
        
        return super().__init__(*args, **kwargs) 

    def select_genomes(self):
        if self._selection_method == 'top':
            self._population = self._population.retrieve_best()

    def fit(self,verbose = 1):
        """
        Main training loop.
        """
        t = 0
        while t < self._max_iterations:          
            self._population.evaluate_population(self._target_genotype)
            if self._population.max_fitness == 1.0:
                break
            else:

                self._population.mutate_genotypes(self._mutation_probability)
                self._population.cross_genomes(crossover_prob=self._crossover_probability)
            t += 1

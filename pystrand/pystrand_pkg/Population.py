import numpy as np
from pystrand_pkg.Genotype import Genotype


class Population(object):
    """
    Collection of individual genotypes.
    Provides facilities for working with multiple genotypes at the same time.
    """
    _individuals = np.array([])

    def __init__(self, 
                 pop_size, 
                 genome_shapes, 
                 random_init = None, 
                 gene_vals = None, 
                 seed = None, 
                 default_genome = None, 
                 *args, **kwargs):

        if type(genome_shapes) is tuple:
            genome_shapes = [genome_shapes for i in range(pop_size)]

        self._individuals = np.array([Genotype(shape,
                                          random_init,
                                          gene_vals,
                                          seed,
                                          default_genome) 
                                 for shape, i in zip(genome_shapes, range(pop_size))])

        return super().__init__(*args, **kwargs)

    def mutate_genotypes(self, mutation_prob = 0.01):
        for genotype in _individuals:
            genotype.mutate(mutation_prob)
    
    def cross_genomes(self, secondary_population = None):

        if secondary_population is None:
            for genotype in _individuals:
                pass
        else:
            pass

    @property
    def population_size(self):
        return self._individuals.size

    @property
    def individuals(self):
        return self._individuals

    @property
    def avg_fitness(self):
        return np.average([genotype.fitness for genotype in _individuals])

    @property
    def max_fitness(self):
        return np.max([genotype.fitness for genotype in _individuals])

    @property
    def min_fitness(self):
        return np.min([genotype.fitness for genotype in _individuals])
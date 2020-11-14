import numpy as np
from pystrand import Genotype


class Population(object):
    """
    Collection of individual genotypes.
    Provides facilities for working with multiple genotypes at the same time.
    """
    
    _dtype = []
    _individuals = np.array([], dtype = _dtype)
    _genome_shapes = np.array([])
    _gene_values = np.array([])

    def __init__(self, 
                 pop_size, 
                 genome_shapes, 
                 random_init = None, 
                 gene_vals = None, 
                 seed = None, 
                 default_genome = None, 
                 *args, **kwargs):

        self._dtype = [('fitness', float), ('genotype', np.object)]
        self._gene_values = gene_vals

        if type(genome_shapes) is tuple:
            self._genome_shapes = [genome_shapes for i in range(pop_size)]
        
        self._individuals = np.array([(0.0,
                                       Genotype(
                                          shape,
                                          random_init,
                                          gene_vals,
                                          seed,
                                          default_genome))
                                 for shape, i in zip(self._genome_shapes, range(pop_size))], dtype=self._dtype)

        return super().__init__(*args, **kwargs)

    def replace_individuals(self, individuals):
        self._individuals = individuals

    def expand_population(self, target_pop_size, strategy = 'clone'):

        if strategy == 'clone':
                self._individuals = np.random.choice(self._individuals, target_pop_size)

        elif strategy == 'random':            
            new_individuals = np.array(
                    [(0.0, Genotype(
                        shape,
                        random_init,
                        gene_vals,
                        seed,
                        default_genome))
                    for shape, i in zip(self._genome_shapes, range(target_pop_size-self.population_size))], 
                    dtype=self._dtype)

            self._individuals = np.append(self._individuals, new_individuals)

    def mutate_genotypes(self, mutation_prob = 0.01):
        for individual in _individuals:
            individual.genotype.mutate(mutation_prob)
    
    def cross_genomes(self, secondary_population = None, crossover_prob = 0.0):
        if secondary_population is None:
            secondary_population = np.array([evaluated_individual.genotype for evaluated_individual in self._individuals])
        
        for individual in self._individuals:
                individual.genotype.crossover(secondary_population)

    def evaluate_individual(self, individual, target):
        pass

    def evaluate_population(self, target = None):
        if target == None:
            self._individuals = [individual.fitness(0.0) for individual in self._individuals]
        else:
            self._individuals = [evaluate_individual(individual.genotype, target) for individual in self._individuals]

    def retrieve_best(self, n = 1):
        return np.sort(self._individuals, order='fitness')[:n]

    @property
    def population_size(self):
        return self._individuals.size

    @property
    def genome_shapes(self):
        return self._genome_shapes
    
    @property
    def gene_values(self):
        return self._gene_values

    @property
    def individuals(self):
        return self._individuals

    @property
    def avg_fitness(self):
        return np.average([genotype.fitness for genotype in self._individuals])

    @property
    def max_fitness(self):
        return np.max([genotype.fitness for genotype in self._individuals])

    @property
    def min_fitness(self):
        return np.min([genotype.fitness for genotype in self._individuals])

    @property
    def fitness_std(self):
        return np.std([genotype.fitness for genotype in self._individuals])
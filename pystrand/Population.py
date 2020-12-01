import numpy as np
from pystrand import Genotype

class Population(object):
    """
    Collection of individual genotypes.
    Provides facilities for working with multiple genotypes at the same time.
    """
    
    _dtype = np.dtype([('fitness', float), ('genotype', np.object)])
    _individuals = np.array([], dtype = _dtype)
    _genome_shapes = np.array([])
    _gene_values = np.array([])
    _random_init = False
    _seed = 0
    _default_genome = None

    def __init__(self, 
                 pop_size, 
                 genome_shapes, 
                 random_init = None, 
                 gene_vals = None, 
                 seed = None, 
                 default_genome = None,
                 seed_individuals = None,
                 *args,
                 **kwargs):
        """

        Arguments:
        pop_size -- number of individuals in given population
        genome_shapes -- shapes of individual genomes as numpy arrays
        random_init -- if the genomes are supposed to be randomized
        gene_vals -- possible values of genes for given population
        seed -- 
        default_genome -- used as genome for entire population, 
                          if random_init = False
        seed_individuals -- numpy array of evaluated inidividuals
        individual_dtype -- numpy dtype defined by Population class

        New individuals are not generated if seed_individuals isn't None.
        """
        self._gene_values = gene_vals
        
        if type(genome_shapes) is tuple:
            self._genome_shapes = [genome_shapes for i in range(pop_size)]
        elif type(genome_shapes) is list:
            self._genome_shapes = genome_shapes        
        
        if seed_individuals is not None:
            self._individuals = seed_individuals
        else:
            self._individuals = np.array([(0.0,
                        Genotype(
                            shape,
                            random_init,
                            gene_vals,
                            seed,
                            default_genome))
                    for shape, i in zip(self._genome_shapes, range(pop_size))], dtype=self._dtype)

        super().__init__(*args, **kwargs)

    def replace_individuals(self, individuals):
        self._individuals = individuals

    def expand_population(
        self, 
        target_pop_size, 
        strategy = 'clone'):

        """
        Increases number of indivuals in given population.
        
        Arguments:

        target_pop_size -- number of individuals we want in population.
        strategy -- how are the new individuals created.
                    Two available options are 'clone' and 'random'.
                    The 'clone' strategy selects random existing individuals,
                    while the 'random' strategy generates new ones.
        """
        if strategy == 'clone':
                self._individuals = np.random.choice(self._individuals, target_pop_size)

        elif strategy == 'random':            
            new_individuals = np.array(
                    [(0.0, Genotype(
                        shape,
                        self._random_init,
                        self._gene_values,
                        self._seed,
                        self._default_genome))
                    for shape, i in zip(self._genome_shapes, range(target_pop_size-self.population_size))], 
                    dtype=self._dtype)

            self._individuals = np.append(self._individuals, new_individuals)

    def mutate_genotypes(self, mutation_prob = 0.01):
        """
        Applies mutation operator to individuals with given probability.
        """
        for individual in self._individuals:
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
            self._individuals = [self.evaluate_individual(individual.genotype, target) for individual in self._individuals]

    def retrieve_best(self, n = 1):
        """
        'n' individuals with highest value of fitness are retrieved.
        """
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
        return np.max([genotype['fitness'] for genotype in self._individuals])

    @property
    def min_fitness(self):
        return np.min([genotype.fitness for genotype in self._individuals])

    @property
    def fitness_std(self):
        return np.std([genotype.fitness for genotype in self._individuals])
        
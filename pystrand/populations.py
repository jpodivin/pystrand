import numpy as np
from pystrand.genotypes import Genotype
from pystrand.mutations import PointMutation


class BasePopulation:
    """
    Collection of individual genotypes.
    Provides facilities for working with multiple genotypes at the same time.

    Parameters
    ----------
        pop_size : number of individuals in given population
        genome_shapes : shapes of individual genomes as numpy arrays
        random_init : if the genomes are supposed to be randomized
        gene_vals : possible values of genes for given population
        seed :
        default_genome : used as genome for entire population,
                            if random_init = False
        seed_individuals : numpy array of evaluated inidividuals
        individual_dtype : numpy dtype defined by Population class

    """

    def __init__(self,
                 pop_size,
                 genome_shapes,
                 random_init=None,
                 gene_vals=None,
                 seed=None,
                 default_genome=None,
                 seed_individuals=None,
                 **kwargs):
        """
        New individuals are not generated if seed_individuals isn't None.
        """
        self._dtype = np.dtype([('fitness', 'd'), ('genotype', 'O')])
        self._gene_values = gene_vals
        self._random_init = random_init
        self._seed = seed
        self._default_genome = default_genome

        if isinstance(genome_shapes, tuple):
            self._genome_shapes = [genome_shapes for i in range(pop_size)]
        elif isinstance(genome_shapes, list):
            self._genome_shapes = genome_shapes

        if seed_individuals is not None:
            self._individuals = seed_individuals
        else:
            self._individuals = np.array(
                [
                    (
                        0.0,
                        Genotype(
                            shape,
                            random_init,
                            gene_vals,
                            seed,
                            default_genome
                            ))
                    for shape, i in zip(self._genome_shapes, range(pop_size))],
                dtype=self._dtype)

        super().__init__(**kwargs)

    def replace_individuals(self, new_individuals):
        """
        Replaces existing individuals managed by population with 'new_individuals'.

        Raises:
            TypeError if new_individuals isn't numpy array of required dtype.
        """
        if not isinstance(new_individuals, np.ndarray):
            raise TypeError()
        if new_individuals.dtype.type is not self._dtype.type:
            raise TypeError("Invalid dtype of new_individuals ndarrays")
        self._individuals = new_individuals

    def expand_population(
            self,
            target_pop_size,
            strategy='clone'):

        """
        Increases number of indivuals in given population.

        Parameters
        ----------
        target_pop_size : number of individuals we want in population.
        strategy : how are the new individuals created.
                    Two available options are 'clone' and 'random'.
                    The 'clone' strategy selects random existing individuals,
                    while the 'random' strategy generates new ones.
        """
        size_difference = target_pop_size-self.population_size

        if strategy == 'clone':
            new_individuals = np.random.choice(self._individuals, size_difference)
            for individual in new_individuals:
                individual['genotype'] = individual['genotype'].clone()
                individual['genotype'].protected = False

        elif strategy == 'random':
            new_individuals = []
            for _ in range(size_difference):
                new_individuals += [
                    (
                        0.0,
                        Genotype(
                            self.genome_shapes[0],
                            self._random_init,
                            self._gene_values,
                            self._seed,
                            self._default_genome
                            )
                        )]

            new_individuals = np.array(
                new_individuals,
                dtype=self._dtype)

        self._individuals = np.append(self._individuals, new_individuals)

    def mutate_genotypes(self, mutation_op=PointMutation(0.01)):
        """
        Applies mutation operator to individuals with given probability.
        """

        for genotype in self._individuals['genotype']:
            if not genotype.protected:
                genotype.mutate(mutation_op)

    def cross_genomes(
            self,
            secondary_population=None,
            crossover_prob=0.0):
        """
        Crosses genomes of inidividuals with those in 'secondary_population'.

        Parameters
        ----------
            secondary_population :
            crossover_prob :
        """
        if secondary_population is None:
            secondary_population = self._individuals['genotype']
        for individual in self._individuals['genotype']:
            if not individual.protected:
                if np.random.random_sample(1) < crossover_prob:
                    individual = individual.crossover(np.random.choice(secondary_population))

    def retrieve_best(self, size=1):
        """
        'n' individuals with highest value of fitness are retrieved.
        Genotype objects don't support comparison, individuals can't be sorted directly.
        """
        indices = np.argsort(self._individuals['fitness'])[-size:]

        #Ugly but it works.
        return np.array(
            [(fitness, genotype.clone()) for fitness, genotype in self._individuals[indices]],
            dtype=self._dtype)

    def append_individuals(self, new_individuals):
        """
        Appends array of 'new_individuals' to existing individuals managed by Population.

        Raises:
            TypeError if new_individuals isn't numpy array of required dtype.
        """
        if not isinstance(new_individuals, np.ndarray) or new_individuals.dtype.type is not self._dtype.type:
            raise TypeError()

        self._individuals = np.append(self._individuals, new_individuals)

    #Properties for easier retrieval of frequently used values.
    @property
    def population_size(self):
        """
        Return size of _individuals ndarray as an integer.
        """
        return self._individuals.size

    @property
    def genome_shapes(self):
        """
        Return _genome_shapes list.
        """
        return self._genome_shapes

    @property
    def gene_values(self):
        """
        Return _gene_values.
        """
        return self._gene_values

    @property
    def individuals(self):
        """
        Return _individuals ndarray.
        """
        return self._individuals

    @property
    def avg_fitness(self):
        """
        Return average fitness as float.
        """
        return np.average([genotype['fitness'] for genotype in self._individuals])

    @property
    def max_fitness(self):
        """
        Return max fitness as float.
        """
        return np.max([genotype['fitness'] for genotype in self._individuals])

    @property
    def min_fitness(self):
        """
        Return min fitness as float.
        """
        return np.min([genotype['fitness'] for genotype in self._individuals])

    @property
    def fitness_std(self):
        """
        Return standard deviation of fitness as float.
        """
        return np.std([genotype['fitness'] for genotype in self._individuals])

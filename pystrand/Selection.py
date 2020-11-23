import numpy as np
from pystrand import Genotype, Population

class Selection(object):
    """
    Base selection class.
    """
    _name = ""

    def __init__(self, 
                *args, 
                **kwargs):
        self._name = kwargs.get("name", "Selection")
        
    def __get_selected_population__(
        self, 
        selected_individuals, 
        population_size, 
        genome_shapes, 
        gene_values,
        individual_dtype):

        selected_individuals = np.array(
            selected_individuals,
            dtype=individual_dtype
            )

        selected_population = Population(
            population_size, 
            genome_shapes,
            gene_vals = gene_values,
            seed_individuals = selected_individuals)

        return selected_population

    def __select__(self, population):
        selected_individuals = []

        for fitness, individual in population.individuals:        
            selected_individuals.append((fitness, individual))

        return self.__get_selected_population__(
            selected_individuals, 
            population.population_size, 
            population.genome_shapes, 
            population.gene_values,
            population.individuals.dtype
            )

class RandomSelection(Selection):
    """
    Randomly selects a fraction of individuals in given population.
    The selection probability is given as argument. 
    """

    _selection_prob = 0.0
    _rng = None

    def __init__(self,
                selection_prob,
                *args,
                **kwargs):

        self._selection_prob = selection_prob
        self._rng = np.random.default_rng()

        super().__init__(args, kwargs)

    def __select__(self, population):

        selected_individuals = []

        for fitness, individual in population.individuals:
            if self._rng.random() < self._selection_prob:
                selected_individuals.append((fitness, individual))

        return self.__get_selected_population__(
            selected_individuals, 
            population.population_size, 
            population.genome_shapes, 
            population.gene_values,
            population.individuals.dtype
            )

class RouletteSelection(Selection):

    def __init__(self, 
                *args,
                **kwargs):

        super().__init__(args, kwargs)

    def __select__(self, population):
        selected_individuals = []

        return self.__get_selected_population__(
            selected_individuals, 
            population.population_size, 
            population.genome_shapes, 
            population.gene_values,
            population.individuals.dtype
            )
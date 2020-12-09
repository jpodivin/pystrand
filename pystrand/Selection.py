import numpy as np
from pystrand import Genotype, Population

class Selection(object):
    """
    Base selection class.
    """
    _name = ""
    _rng = None
    
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
        """
        Creates new Population object from selected_individuals. 
        
        Arguments:
            selected_individuals -- seed individuals forming the population
            population_size -- number of individuals in given population
            genome_shapes -- shapes of individual genomes as numpy arrays
            gene_values -- possible values of genes for given population
            individual_dtype -- numpy dtype defined by Population class
        """
        selected_individuals = np.array(
            selected_individuals,
            dtype=individual_dtype            
            ).flatten()

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
            
    def select(self, population):

        return self.__select__(population)

class RandomSelection(Selection):
    """
    Randomly selects a fraction of individuals in given population.
    The selection probability is given as argument.     
    """

    _selection_prob = 0.0    

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
    """
    Naive implementation of Roulette selection (or Fitness proportionate selection).
    Checks for case of maximum fitness = 0 and assignes equal probability to all individuals.

    """
    _selected_population_fraction = 0

    def __init__(self,
                selected_population_fraction,
                *args,
                **kwargs):
        
        self._selected_population_fraction = selected_population_fraction
        self._rng = np.random.default_rng()

        super().__init__(args, kwargs)

    def __select__(self, population):
        selected_individuals = []
        n_selected = int(population.population_size*self._selected_population_fraction)
        probs = population.individuals["fitness"]

        if probs.max() > 0.0:
            scaling = lambda x: x / np.sum(probs)
            probs = np.apply_along_axis(scaling, 0, probs)
        else:
            probs = np.full(probs.shape, 1.0/probs.size)

        selected_individuals = self._rng.choice(
                population.individuals, 
                size = n_selected,
                p = probs)
                
        return self.__get_selected_population__(
            selected_individuals, 
            population.population_size, 
            population.genome_shapes, 
            population.gene_values,
            population._dtype
            )
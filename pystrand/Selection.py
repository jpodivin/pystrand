import numpy as np
from pystrand import Genotype
from pystrand import Population

class Selection(object):
    
    _criterion = None

    def __init__(self, 
                *args, 
                **kwargs):

        self._criterion = kwargs.get("criterion", lambda inividual: True)    

    def __select__(self, population):
        individuals = population.individuals
        filtered_individuals = individuals
        population.replace_individuals(filtered_individuals)
        return population

class RandomSelection(Selection):

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

        selected_individuals = np.array(
            selected_individuals,
            dtype=population.individuals.dtype
            )

        selected_population = Population(
            population.population_size, 
            population.genome_shapes,
            gene_vals = population.gene_values,
            seed_individuals = selected_individuals)

        return selected_population
                
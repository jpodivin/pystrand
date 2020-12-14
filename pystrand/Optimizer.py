import numpy as np
from pystrand import Genotype, Population, RouletteSelection, Selection

class Optimizer(object):
    """Base optimizer class.
    
    Raises:
        TypeError: if supplied wrong selection method type 
    """
    _population = None
    _max_iterations = 0
    _fitness_function = lambda x: 0.0
    _mutation_probability = 0.0
    _crossover_probability = 0.0
    _selection_method = None
    _elitism = 0

    def __init__(self, 
                 fitness_function, 
                 max_iterations,
                 population,
                 mutation_prob = 0.001,
                 crossover_prob = 0.0,
                 selection_method = 'roulette',
                 selected_fraction = 0.1,                 
                 outfile='', 
                 *args, 
                 **kwargs):
        """
        Arguments:
            fitness_function -- provides mapping from genotype to fitness value, [0, 1] 
            max_iterations -- 
            population -- Seed population, can include known sub-optimal solutions.
            mutation_prob -- 0.001 by default
            crossover_prob -- 0.0 by default, no crossover will take place
            selection_method -- 
            selected_fraction --            
            outfile --
        Raises:

        """
        self._fitness_function = fitness_function
        self._mutation_probability = mutation_prob
        self._crossover_probability = crossover_prob
        
        if type(selection_method) is str:
            if selection_method == 'roulette':
                self._selection_method = RouletteSelection(selected_fraction)
        elif type(selection_method) is Selection:
            self._selection_method = selection_method
        else:
            raise TypeError(
                'Invalid selection type.',
                type(selection_method)
                )

        self._population = population 

        self._max_iterations = max_iterations
        
        self._elitism = int(
            kwargs.get('elitism', 0.0) * selected_fraction * population.population_size
            )

        super().__init__(*args, **kwargs) 

    def evaluate_individual(self, individual):
        
        return self._fitness_function(individual)

    def evaluate_population(self):
        evaluated_individuals = self._population.individuals
    
        for individual in evaluated_individuals:
            individual['fitness'] = self.evaluate_individual(
                            individual['genotype'].genome
                            )
            
        self._population.replace_individuals(evaluated_individuals)

    def select_genomes(self):
        new_population = self._selection_method.select(self._population)
        new_population.expand_population(self._population.population_size)
        self._population = new_population

    def fit(self, verbose = 1):
        """
        Main training loop.
        """
        history = {
            "iteration" : [],
            "max_fitness" : [],
            "min_fitness" : [],
            "fitness_avg" : [],
            "fitness_std" : []
            }

        t = 0
        
        while t < self._max_iterations:
            self.evaluate_population()

            history["iteration"].append(t)
            history["max_fitness"].append(self._population.max_fitness)
            history["min_fitness"].append(self._population.min_fitness)
            history["fitness_avg"].append(self._population.avg_fitness)
            history["fitness_std"].append(self._population.fitness_std)

            if verbose > 0:
                print(" // ".join(
                    [key + ": " + str(record[-1]) for key, record in history.items()]
                    )
                )

            if self._population.max_fitness == 1.0:
                break
            else:
                self.select_genomes()

                if self._elitism > 0.0:
                    holdout = self._population.retrieve_best(self._elitism)

                self._population.mutate_genotypes(self._mutation_probability)

                if self._crossover_probability > 0.0:
                    self._population.cross_genomes(
                        crossover_prob = self._crossover_probability,
                        secondary_population = self._population.individuals['genotype'])
            t += 1

        return history

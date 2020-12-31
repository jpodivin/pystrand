import numpy as np
import multiprocessing as mp
from pystrand import Genotype, Population, RouletteSelection, ElitismSelection, Selection

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
    _selection_methods = None

    def __init__(self, 
                 fitness_function, 
                 max_iterations,
                 population,
                 mutation_prob = 0.001,
                 crossover_prob = 0.0,
                 selection_methods = 'roulette',
                 selected_fraction = 0.1,
                 outfile='', 
                 parallelize = False,
                 *args, 
                 **kwargs):
        """
        Arguments:
            fitness_function -- provides mapping from genotype to fitness value, [0, 1] 
            max_iterations -- 
            population -- Seed population, can include known sub-optimal solutions.
            mutation_prob -- 0.001 by default
            crossover_prob -- 0.0 by default, no crossover will take place
            selection_methods -- 
            selected_fraction -- 
            outfile --
        Raises:
            TypeError:
        """
        self._fitness_function = fitness_function
        self._mutation_probability = mutation_prob
        self._crossover_probability = crossover_prob 
        self._selection_methods = []
        self._parallelize = parallelize
        
        #First we turn selection_methods into list, in case it isn't.
        if type(selection_methods) is not list:
            selection_methods = [selection_methods]        
        """        
        For each element in list of sleection methods we check the type.
        Only Selection and string are accepted, other types raise TypeError.
        The strings must be reckognized as names of algorithm, 
        any other string will result in ValueError.
        """

        for selection_method in selection_methods:
            if type(selection_method) is str:
                if selection_method == 'roulette':
                    self._selection_methods += [RouletteSelection(selected_fraction)]
                elif selection_method == 'elitism':
                    self._selection_methods += [ElitismSelection(selected_fraction)]
                else:
                    raise ValueError(
                    'Unknown selection algorithm name.',
                    selection_method
                    )
            elif type(selection_method) is Selection:
                self._selection_methods += [selection_method]
            else:
                raise TypeError(
                    'Invalid selection type.',
                    type(selection_method)
                    )

        self._population = population 

        self._max_iterations = max_iterations     

    def evaluate_individual(self, individual):
        
        return self._fitness_function(individual)

    def evaluate_population(self):
        
        evaluated_individuals = self._population.individuals
        if self._parallelize:
            with mp.Pool() as worker_pool:
                evaluated_individuals['fitness'] = worker_pool.map(
                    self._fitness_function,
                    evaluated_individuals['genotype'])
        else:
            evaluated_individuals['fitness'] = [
                self._fitness_function(individual) 
                for individual 
                in evaluated_individuals['genotype']
                ]

        self._population.replace_individuals(evaluated_individuals)

    def select_genomes(self):
        new_population = Population(
            0, 
            self._population.genome_shapes,
            self._population.gene_values)

        for selection_method in self._selection_methods:
            new_population.append_individuals(
                selection_method.select(self._population)
                )

        new_population.expand_population(
            self._population.population_size
            )
     
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
        
        best_performers = np.array([])
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
                
                self._population.mutate_genotypes(self._mutation_probability)

                if self._crossover_probability > 0.0:
                    self._population.cross_genomes(
                        crossover_prob = self._crossover_probability
                        )

                if best_performers.size > 0:
                    self._population.append_individuals(best_performers)

            t += 1
            
        return history

import numpy as np
from pystrand import Genotype, Population
from pystrand.Selection import Selection, RouletteSelection

class Optimizer(object):
    """Base optimizer class.
    
    Raises:
        TypeError: if supplied wrong selection method type 
    """
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
                 selection_method = 'roulette',
                 selected_fraction = 0.1,
                 population = None,
                 outfile='', 
                 *args, 
                 **kwargs):
        """
        Arguments:
            target_genotype -- 
            max_iterations -- 
            mutation_prob -- 0.001 by default
            crossover_prob -- 0.0 by default, no crossover will take place
            selection_method -- 
            selected_fraction --
            population -- Seed population, can include known sub-optimal solutions.
                If the population isn't set the parameters are inferred from target_genotype.
            outfile --
        Raises:

        """
        self._target_genotype = target_genotype
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

        if population is not None:
            self._population = population
        else:
            """
            Population parameter inferrence.
            """
            self._population = Population(
                pop_size = np.sum(target_genotype.genome.shape)*10, 
                genome_shapes = target_genotype.genome.shape,
                gene_vals = target_genotype.gene_vals,
                random_init = True
                )

        self._max_iterations = max_iterations
        
        super().__init__(*args, **kwargs) 

    def evaluate_individual(self, individual, target):
        difference = np.sum(np.not_equal(individual, target))
        return 1 - difference/individual.size

    def evaluate_population(self, target = None):
        evaluated_individuals = self._population.individuals
        if target == None:            
            for individual in evaluated_individuals:
                individual['fitness'] = 0.0
        else:
            for individual in evaluated_individuals:
                individual['fitness'] = self.evaluate_individual(
                                individual['genotype'].genome, 
                                target.genome
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
            self.evaluate_population(self._target_genotype)

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
                        crossover_prob = self._crossover_probability,
                        secondary_population = self._population.individuals['genotype'])
            t += 1

        return history

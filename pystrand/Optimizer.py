import numpy as np
from pystrand import Genotype, Population, Selection, RouletteSelection

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
            self._population = Population(
                target_genotype.shape()**2, 
                target_genotype.shape())

        self._max_iterations = max_iterations
        
        super().__init__(*args, **kwargs) 

    def select_genomes(self):
        self._population = self._selection_method.select(self._population)

    def fit(self,verbose = 1):
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
            history["iteration"].append(t)
            history["max_fitness"].append(self._population.max_fitness)
            history["min_fitness"].append(self._population.min_fitness)
            history["fitness_avg"].append(self._population.avg_fitness)
            history["fitness_std"].append(self._population.fitness_std)

            self._population.evaluate_population(self._target_genotype)
            
            if self._population.max_fitness == 1.0:
                break
            else:
                self.select_genomes()
                self._population.mutate_genotypes(self._mutation_probability)
                self._population.cross_genomes(crossover_prob=self._crossover_probability)
            t += 1

        return history

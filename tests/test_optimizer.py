from pystrand import Optimizer, Genotype, Population
import unittest
import numpy as np

target_genotypes_small = [
        np.zeros((10),),
        np.array([i%2 for i in range(10)]),
        np.array([i+1%2 for i in range(10)]),
        np.array([i%3 for i in range(10)])
        ]

target_genotypes_large = [
    np.resize(array, (100,)) for array in target_genotypes_small
    ]

class Optimizer_init_test(unittest.TestCase):

    def test_optimizer_init_small(self):
        """
        Optimizer init test. Only checks genotype preservation and instantiation
        """
        
        for target_genotype in target_genotypes_small:

            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype
                )

            population = Population(
                pop_size = np.sum(target_genotype.shape)*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = target_genotype.gene_vals,
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,
                10,
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5
                )

            self.assertIsInstance(new_optimizer, Optimizer)

            self.assertEqual(new_optimizer._fitness_function, fitness_fn)

    def test_optimizer_init_large(self):
        """
        Optimizer init test. Only checks genotype preservation and instantiation
        """
        
        for target_genotype in target_genotypes_large:

            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype
                )

            population = Population(
                pop_size = np.sum(target_genotype.shape)*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = target_genotype.gene_vals,
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,
                10,
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5
                )

            self.assertIsInstance(new_optimizer, Optimizer)

            self.assertEqual(new_optimizer._fitness_function, fitness_fn)

class Optimizer_Run_test(unittest.TestCase):

    test_runtime_short = 10    
    test_runtime_long = 100
    history_dict_keys = [
        'iteration', 
        'max_fitness', 
        'min_fitness', 
        'fitness_avg', 
        'fitness_std'
        ]
    
    def test_optimizer_run_small(self):
        """
        Short run of basic optimizer with default params and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        
        for target_genotype in target_genotypes_small:
            
            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            population = Population(
                pop_size = np.sum(target_genotype.shape)*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,
                self.test_runtime_short,
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5
                )

            history = new_optimizer.fit(verbose=0)
            
            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys()) 
                and set(history.keys()).issubset(self.history_dict_keys)
                )

            self.assertLessEqual(max(history['iteration']), self.test_runtime_short)

    def test_optimizer_run_large(self):
        """
        Long run of basic optimizer with default params and binary genome.
        10000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        for target_genotype in target_genotypes_large:

            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            population = Population(
                pop_size = np.sum(target_genotype.shape)*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,
                self.test_runtime_short,
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5
                )
                
            history = new_optimizer.fit(verbose=0)
            
            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys()) 
                and set(history.keys()).issubset(self.history_dict_keys)
                )

            self.assertLessEqual(max(history['iteration']), self.test_runtime_long)
    
    def test_optimizer_elitism_small(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        
        for target_genotype in target_genotypes_small[1:]:
        
            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            population = Population(
                pop_size = target_genotype.size*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,                
                self.test_runtime_short,
                selection_methods = 'elitism',
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5
                )

            history = new_optimizer.fit(verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0, 
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness']))

    def test_optimizer_elitism_large(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        
        for target_genotype in target_genotypes_large[1:]:
            
            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            population = Population(
                pop_size = target_genotype.size*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,                
                self.test_runtime_short,
                selection_methods = 'elitism',
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5
                )

            history = new_optimizer.fit(verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0, 
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness']))


    def test_optimizer_combined_small(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        
        for target_genotype in target_genotypes_small[1:]:
        
            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            population = Population(
                pop_size = target_genotype.size*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,
                self.test_runtime_short,
                selection_methods = ['elitism', 'roulette'],
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5            
                )

            history = new_optimizer.fit(verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0, 
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s " %(target_genotype, history['max_fitness']))


    def test_optimizer_combined_large(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        
        for target_genotype in target_genotypes_large[1:]:
            
            def fitness_fn(individual):
                difference = np.sum(np.not_equal(individual, target_genotype))
                return 1 - difference/individual.size

            population = Population(
                pop_size = target_genotype.size*10, 
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True
                )

            new_optimizer = Optimizer(
                fitness_fn,
                self.test_runtime_short,
                selection_methods = ['elitism', 'roulette'],
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5
                )

            history = new_optimizer.fit(verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0, 
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness']))

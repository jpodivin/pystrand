from pystrand import Optimizer, Genotype
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

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype)

            new_optimizer = Optimizer(target_genotype, 10)

            self.assertIsInstance(new_optimizer, Optimizer)

            self.assertEqual(new_optimizer._target_genotype, target_genotype)

    def test_optimizer_init_large(self):
        """
        Optimizer init test. Only checks genotype preservation and instantiation
        """
        
        for target_genotype in target_genotypes_large:

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype
                )

            new_optimizer = Optimizer(target_genotype, 10)

            self.assertIsInstance(new_optimizer, Optimizer)

            self.assertEqual(new_optimizer._target_genotype, target_genotype)

class Optimizer_Run_test(unittest.TestCase):

    test_runtime_short = 1000    
    test_runtime_long = 10000
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

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype)

            new_optimizer = Optimizer(
                target_genotype,
                self.test_runtime_short, 
                mutation_prob = 0.1,
                crossover_prob = 0.5)

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

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype)

            new_optimizer = Optimizer(
                target_genotype,
                self.test_runtime_long, 
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

from pystrand import Optimizer, Genotype
import unittest
import numpy as np

class Optimizer_Small_test(unittest.TestCase):

    test_runtime = 1000    

    target_genotypes = [
            np.array([0 for i in range(10)]),
            np.array([i%2 for i in range(10)])
            ]

    history_dict_keys = [
        'iteration', 
        'max_fitness', 
        'min_fitness', 
        'fitness_avg', 
        'fitness_std'
        ]

    def test_optimizer_init(self):
        """
        Optimizer init test. Only checks genotype preservation and instantiation
        """
        
        for target_genotype in self.target_genotypes:

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype)

            new_optimizer = Optimizer(target_genotype, self.test_runtime)

            self.assertIsInstance(new_optimizer, Optimizer)

            self.assertEqual(new_optimizer._target_genotype, target_genotype)
    
    def test_optimizer_run(self):
        """
        Short run of basic optimizer with default params and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        
        for target_genotype in self.target_genotypes:

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype)

            new_optimizer = Optimizer(
                target_genotype,
                self.test_runtime, 
                mutation_prob = 0.1,
                crossover_prob = 0.1)

            history = new_optimizer.fit()
            
            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys()) 
                and set(history.keys()).issubset(self.history_dict_keys)
                )

            self.assertLessEqual(max(history['iteration']), self.test_runtime)
            
class Optimizer_Large_test(unittest.TestCase):

    test_runtime = 1000

    target_genotypes = [
            np.array([0 for i in range(100)]),
            np.array([i%2 for i in range(100)])
            ]

    history_dict_keys = [
        'iteration', 
        'max_fitness', 
        'min_fitness', 
        'fitness_avg', 
        'fitness_std'
        ]

    def test_optimizer_init(self):
        """
        Optimizer init test. Only checks genotype preservation and instantiation
        """
        
        for target_genotype in self.target_genotypes:

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype)

            new_optimizer = Optimizer(target_genotype, self.test_runtime)

            self.assertIsInstance(new_optimizer, Optimizer)

            self.assertEqual(new_optimizer._target_genotype, target_genotype)
    
    def test_optimizer_run(self):
        """
        Short run of basic optimizer with default params and binary genome.
        10000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """      
        
        for target_genotype in self.target_genotypes:

            target_genotype = Genotype(
                target_genotype.shape, 
                gene_vals=np.unique(target_genotype), 
                default_genome=target_genotype)

            new_optimizer = Optimizer(
                target_genotype,
                self.test_runtime, 
                mutation_prob = 0.1,
                crossover_prob = 0.1
                )

            history = new_optimizer.fit()
            
            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys()) 
                and set(history.keys()).issubset(self.history_dict_keys)
                )

            self.assertLessEqual(max(history['iteration']), self.test_runtime)
            
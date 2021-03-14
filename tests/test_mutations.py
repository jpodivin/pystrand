import unittest
import numpy as np
from pystrand.genotypes import Genotype
from pystrand.mutations import PointMutation, BlockMutation

class TestPointMutation(unittest.TestCase):

    def setUp(self):
        self.test_shapes = [(i, j, k) 
                  for i in range(1, 10) 
                  for j in range(1, 10) 
                  for k in range(1, 10)]

        self.test_gene_vals = [np.ceil(np.random.normal(scale = 10, size = 10)) for i in range(100)]
    
    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for shape in self.test_shapes:
            for gene_vals in self.test_gene_vals:
                genome = Genotype(shape, 
                                random_init = True,
                                gene_vals = gene_vals)

                original_genome = genome.clone()

                genome.mutate(PointMutation(1.0))

                self.assertFalse(np.array_equiv(genome, original_genome))

                self.assertTrue(genome.max() <= gene_vals.max())

                self.assertTrue(genome.min() >= gene_vals.min())

class TestBlockMutation(unittest.TestCase):

    def setUp(self):
        self.test_shapes = [(i, j, k) 
                  for i in range(1, 10) 
                  for j in range(1, 10) 
                  for k in range(1, 10)]

        self.test_gene_vals = [np.ceil(np.random.normal(scale = 10, size = 10)) for i in range(100)]
    
    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for shape in self.test_shapes:
            for gene_vals in self.test_gene_vals:
                genome = Genotype(shape, 
                                random_init = True,
                                gene_vals = gene_vals)

                genome.mutate(BlockMutation(1.0, block_size=10))

                self.assertTrue(genome.max() <= gene_vals.max())

                self.assertTrue(genome.min() >= gene_vals.min())
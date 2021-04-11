import unittest
import numpy as np
from pystrand.genotypes import Genotype
from pystrand.mutations import *


test_genotypes = {}
test_shapes = [
    (i, j, k)
        for i in range(1, 10)
        for j in range(1, 10)
        for k in range(1, 10)]

test_gene_vals = [np.ceil(np.random.normal(scale = 10, size = 10)) for i in range(100)]

for shape in test_shapes:
    for gene_vals in test_gene_vals:
        test_genotypes[id(shape)*id(gene_vals)] = {
            'genotype': Genotype(
                shape,
                random_init=True,
                gene_vals=gene_vals),
            'gene_vals': gene_vals,
            'shape': shape
        }


class TestPointMutation(unittest.TestCase):

    def setUp(self):
        self.test_genotypes = test_genotypes.copy()
        super(TestPointMutation, self).setUp()
    
    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for genome in self.test_genotypes:
            genome = self.test_genotypes[genome]
            altered_genome = genome['genotype'].clone()

            altered_genome.mutate(PointMutation(1.0))

            self.assertFalse(np.array_equiv(genome, altered_genome))

            self.assertTrue(
                genome['genotype'].max() <= genome['gene_vals'].max())

            self.assertTrue(
                genome['genotype'].min() >= genome['gene_vals'].min())


class TestBlockMutation(unittest.TestCase):

    def setUp(self):
        self.test_genotypes = test_genotypes.copy()
        super(TestBlockMutation, self).setUp()
    
    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for genome in self.test_genotypes:
            genome = self.test_genotypes[genome]
            altered_genome = genome['genotype'].clone()

            altered_genome.mutate(BlockMutation(1.0))

            self.assertFalse(np.array_equiv(genome, altered_genome))

            self.assertTrue(
                genome['genotype'].max() <= genome['gene_vals'].max())

            self.assertTrue(
                genome['genotype'].min() >= genome['gene_vals'].min())

class TestPermutationMutation(unittest.TestCase):

    def setUp(self):
        self.test_genotypes = test_genotypes.copy()
        super(TestPermutationMutation, self).setUp()
    
    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for genome in self.test_genotypes:
            genome = self.test_genotypes[genome]
            altered_genome = genome['genotype'].clone()

            altered_genome.mutate(PermutationMutation(1.0))

            self.assertFalse(np.array_equiv(genome, altered_genome))

            self.assertTrue(
                genome['genotype'].max() <= genome['gene_vals'].max())

            self.assertTrue(
                genome['genotype'].min() >= genome['gene_vals'].min())

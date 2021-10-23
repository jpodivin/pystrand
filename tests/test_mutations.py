import unittest
import numpy as np
from pystrand.genotypes import Genotype
import pystrand.operators.mutations as mut

test_genotypes = {}
test_shapes = [
    (i, j, k)
        for i in range(1, 10)
        for j in range(1, 10)
        for k in range(1, 10)
    ]

test_gene_vals = [np.ceil(np.random.normal(scale = 10, size = 10)) for i in range(10)]

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
        self.test_genotypes = test_genotypes
        super(TestPointMutation, self).setUp()

    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for genome in self.test_genotypes:
            genome = self.test_genotypes[genome]
            altered_genome = genome['genotype'].copy()

            altered_genome.mutate(mut.PointMutation(1.0))

            self.assertFalse(np.array_equiv(genome, altered_genome))

            self.assertTrue(
                genome['genotype'].max() <= genome['gene_vals'].max())

            self.assertTrue(
                genome['genotype'].min() >= genome['gene_vals'].min())


class TestBlockMutation(unittest.TestCase):

    def setUp(self):
        self.test_genotypes = test_genotypes
        super(TestBlockMutation, self).setUp()

    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for genome in self.test_genotypes:
            genome = self.test_genotypes[genome]
            altered_genome = genome['genotype'].copy()

            altered_genome.mutate(mut.BlockMutation(1.0))

            self.assertFalse(np.array_equiv(genome, altered_genome))

            self.assertTrue(
                genome['genotype'].max() <= genome['gene_vals'].max())

            self.assertTrue(
                genome['genotype'].min() >= genome['gene_vals'].min())

class TestPermutationMutation(unittest.TestCase):

    def setUp(self):
        self.test_genotypes = test_genotypes
        super(TestPermutationMutation, self).setUp()

    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for genome in self.test_genotypes:
            genome = self.test_genotypes[genome]
            altered_genome = genome['genotype'].copy()

            altered_genome.mutate(mut.PermutationMutation(1.0))

            self.assertFalse(np.array_equiv(genome, altered_genome))

            self.assertTrue(
                genome['genotype'].max() <= genome['gene_vals'].max())

            self.assertTrue(
                genome['genotype'].min() >= genome['gene_vals'].min())

class TestShiftMutation(unittest.TestCase):

    def setUp(self):
        self.test_genotypes = test_genotypes
        super(TestShiftMutation, self).setUp()

    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for genome in self.test_genotypes:
            genome = self.test_genotypes[genome]
            altered_genome = genome['genotype'].copy()

            altered_genome.mutate(mut.ShiftMutation(1.0))

            self.assertFalse(np.array_equiv(genome, altered_genome))

            self.assertTrue(
                genome['genotype'].max() <= genome['gene_vals'].max())

            self.assertTrue(
                genome['genotype'].min() >= genome['gene_vals'].min())

import unittest
import numpy as np
from pystrand.genotypes import Genotype
import pystrand.operators.mutations as mut
"""Tests for mutation operators.
All tests use the same Genotype definition array,
so there is a considerable degree of code duplication.
Nevertheless, it is better to keep things explicit.
"""

class TestMutation(unittest.TestCase):
    """Base class of mutation operator tests.
    Doesn't include any tests by itself, only defines
    shared setUp function for all the others.
    """

    def setUp(self):

        self.test_genotypes = {}
        test_shapes = [
            (i, j, k)
                for i in range(1, 10, 2)
                for j in range(1, 10, 2)
                for k in range(1, 10, 2)]

        test_gene_vals = [
            np.ceil(np.random.normal(scale = 10, size = 10)) for i in range(10)]

        for shape in test_shapes:
            for gene_vals in test_gene_vals:
                self.test_genotypes[id(shape)*id(gene_vals)] = {
                    'genotype': Genotype(
                        shape,
                        random_init=True,
                        gene_vals=gene_vals),
                    'gene_vals': gene_vals,
                    'shape': shape
                }


class TestPointMutation(TestMutation):
    """Tests of the point mutation operator.
    """

    def test_genotype_mutation_bounds(self):
        """Checks operation of mutation operator.
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


class TestBlockMutation(TestMutation):
    """Tests of the block mutation operator.
    """

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

class TestPermutationMutation(TestMutation):
    """Tests of the Permutation mutation operator.
    """

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

class TestShiftMutation(TestMutation):
    """Tests of the Shift mutation operator
    """

    def test_genotype_mutation_bounds(self):
        """Checks operation of mutation operator.
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

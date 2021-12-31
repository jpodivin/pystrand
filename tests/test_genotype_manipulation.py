import unittest
import numpy as np
from pystrand.genotypes import Genotype


class Test_genotype_manipulation(unittest.TestCase):

    def setUp(self):
        self.test_shapes = [(i, j, k)
                  for i in range(1, 10)
                  for j in range(1, 10)
                  for k in range(1, 10)]

        self.test_gene_vals = [np.ceil(np.random.normal(scale = 10, size = 10)) for i in range(100)]

    def test_genotype_initiation_shape(self):
        """
        Checks shapes of genomes.
        """
        for shape in self.test_shapes:
            genome = Genotype(shape)
            self.assertIsInstance(genome, Genotype)
            self.assertEqual(genome.shape, shape)

    def test_genotype_initiation_zeros(self):
        """
        Checks properties of zero initialized genomes.
        """
        for shape in self.test_shapes:
            genome = Genotype(shape)
            self.assertTrue(np.array_equiv(genome, np.zeros(shape)))

    def test_genotype_initiation_random(self):
        """
        Checks properties of randomly initialized genomes.
        """
        for shape in self.test_shapes:
            for gene_vals in self.test_gene_vals:
                genome = Genotype(shape,
                                random_init = True,
                                gene_vals = gene_vals)

                self.assertTrue(genome.max() <= gene_vals.max())

                self.assertTrue(genome.min() >= gene_vals.min())

    def test_genotype_crossover_binary(self):

        for shape in self.test_shapes:
            mask_size = np.product(shape)
            # Boolean mask that will efectivelly split the chromosome in two.
            crossover_mask = np.reshape(
                np.array([True if i < mask_size//2 else False for i in range(mask_size)]),
                shape)

            parent_a = Genotype(
                            shape,
                            default_genome=np.ones(shape))
            parent_b = Genotype(
                            shape,
                            default_genome=np.zeros(shape))

            descendant_a = parent_a.crossover(parent_b, crossover_mask)
            descendant_b = parent_b.crossover(parent_a, crossover_mask)

            self.assertFalse(np.array_equiv(descendant_b, parent_a))
            self.assertFalse(np.array_equiv(descendant_a, parent_b))

            if mask_size > 1:
                self.assertFalse(np.array_equiv(descendant_b, parent_b))
                self.assertFalse(np.array_equiv(descendant_a, parent_a))

if __name__ == '__main__':
    unittest.main()

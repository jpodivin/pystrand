import unittest
import numpy as np
from pystrand_pkg.Genotype import Genotype

class Test_test_genome_manipulation(unittest.TestCase):

    test_shapes = [(i, j, k) 
                  for i in range(1, 10) 
                  for j in range(1, 10) 
                  for k in range(1, 10)]

    test_gene_vals = [np.random.normal(scale = 10, size = 10) for i in range(100)]

    def test_genotype_initiation_shape(self):        
        """
        Checks shapes of genomes.
        """
        for shape in self.test_shapes:
            genome = Genotype(shape)
            self.assertIsInstance(genome, Genotype)
            self.assertEqual(genome.genotype_shape, shape)

    def test_genotype_initiation_zeros(self):
        """
        Checks properties of zero initialized genomes.
        """
        for shape in self.test_shapes:
            genome = Genotype(shape)
            self.assertTrue(np.array_equiv(genome.genome, np.zeros(shape)))

    def test_genotype_initiation_random(self):
        """
        Checks properties of randomly initialized genomes.
        """
        for shape in self.test_shapes:
            for gene_vals in self.test_gene_vals:
                genome = Genotype(shape, 
                                random_init = True,
                                gene_vals = gene_vals)

                self.assertTrue(genome.genome.max() <= gene_vals.max())

                self.assertTrue(genome.genome.min() >= gene_vals.min())

    def test_genotype_mutation_bounds(self):
        """
        Checks operation of mutation operator.
        """
        for shape in self.test_shapes:
            for gene_vals in self.test_gene_vals:
                genome = Genotype(shape, 
                                random_init = False,
                                gene_vals = gene_vals)

                original_genome = genome.clone()
                genome.mutate(1.0)

                self.assertFalse(np.array_equiv(genome.genome, original_genome.genome))

                self.assertTrue(genome.genome.max() <= gene_vals.max())

                self.assertTrue(genome.genome.min() >= gene_vals.min())


if __name__ == '__main__':
    unittest.main()

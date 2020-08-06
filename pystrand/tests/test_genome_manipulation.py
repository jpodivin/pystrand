import unittest
import numpy as np
from pystrand_pkg.Genome import Genome

class Test_test_genome_manipulation(unittest.TestCase):

    test_shapes = [(i, j, k) 
                  for i in range(10) 
                  for j in range(10) 
                  for k in range(10)]

    def test_genotype_initiation_shape(self):        
        """
        Checks shapes of genomes.
        """
        for shape in self.test_shapes:
            genome = Genome(shape)
            self.assertIsInstance(genome, Genome)
            self.assertEqual(genome.get_genotype_shape(), shape)

    def test_genotype_initiation_zeros(self):
        """
        Checks properties of zero initialized genomes.
        """
        for shape in self.test_shapes:
            genome = Genome(shape)
            self.assertTrue(np.array_equiv(genome.get_genotype(), np.zeros(shape)))

    def test_genotype_initiation_random(self):
        """
        Checks properties of randomly initialized genomes.
        First 1000 shape settings aren't used because the resulting genome is length 0.
        """
        for shape in self.test_shapes[1000:]:
            genome = Genome(shape, 
                            random_init = True)
            self.assertFalse(np.array_equiv(genome.get_genotype(), np.zeros(shape)))

            self.assertTrue(genome.get_genotype().max() <= 1.0)

            self.assertTrue(genome.get_genotype().min() >= 0.0)

if __name__ == '__main__':
    unittest.main()

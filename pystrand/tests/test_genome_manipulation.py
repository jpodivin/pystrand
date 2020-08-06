import unittest
import numpy as np
from pystrand_pkg import Genome

class Test_test_genome_manipulation(unittest.TestCase):

    test_shapes = [(i, j, k) 
                  for i in range(10) 
                  for j in range(10) 
                  for k in range(10)]

    def test_genotype_initiation_shape(self):        

        for shape in self.test_shapes:
            genome = Genome.Genome(shape)
            self.assertIs(type(genome), Genome.Genome)
            self.assertEqual(genome.get_genotype_shape(), shape)

    def test_genotype_initiation_zeros(self):

        for shape in self.test_shapes:
            genome = Genome.Genome(shape)
            self.assertTrue(np.array_equiv(genome.get_genotype(), np.zeros(shape)))

    def test_genotype_initiation_random(self):
        for shape in self.test_shapes:
            genome = Genome.Genome(shape)
            self.assertTrue(np.array_equiv(genome.get_genotype(), np.zeros(shape)))

if __name__ == '__main__':
    unittest.main()

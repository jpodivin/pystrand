import unittest
from pystrand_pkg.Population import Population
from pystrand_pkg.Genotype import Genotype

class Test_population(unittest.TestCase):
    pop_sizes = [i for i in range(100)]

    def test_individual_generation(self):
        for pop_size in self.pop_sizes:
            population = Population(pop_size, 
                                    (100, 1))

            self.assertIsInstance(population, Population)
            self.assertTrue(population.population_size == pop_size)

            if pop_size > 0:
                for individual in population.individuals:
                    self.assertIsInstance(individual['genotype'], Genotype)

    def test_evaluation_permanence(self):
        for pop_size in self.pop_sizes:

            population = Population(pop_size,
                                    (100, 1))

            for individual_fitness in population.individuals['fitness']:
                self.assertTrue(individual_fitness == 0.0)

if __name__ == '__main__':
    unittest.main()

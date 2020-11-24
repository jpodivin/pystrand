import unittest
from pystrand import Selection, Population, RandomSelection, RouletteSelection

class Dummy_Selection_Test(unittest.TestCase):
    test_population = Population(100, (100, 1))

    def test_selection_init(self):
        selection = Selection()
        self.assertIsInstance(selection, Selection)

    def test_selection(self):
        selection = Selection()
        
        selected_population = selection.__select__(self.test_population)

        self.assertEqual(
            selected_population.population_size, 
            self.test_population.population_size
            )

class Random_Selection_Test(unittest.TestCase):

    test_population = Population(1000000, (100, 1))
    selection_probabilities = [0.1, 0.5, 0.9]

    def test_selection_init(self):
        selection = RandomSelection(0.5)
        self.assertIsInstance(selection, RandomSelection)

    def test_selection(self):
        for selection_probability in self.selection_probabilities:
            selection = RandomSelection(selection_probability)
            selected_population = selection.__select__(self.test_population)
            
            self.assertNotEqual(
                selected_population.population_size,
                self.test_population.population_size
                )

            self.assertAlmostEqual(
                selected_population.population_size/self.test_population.population_size,
                selection_probability,
                places=1
                )

class Roulette_Selection_Test(unittest.TestCase):

    test_population = Population(1000, (100, 1))
    population_fractions = [0.1, 0.5, 0.9]
    max_fitness_values = [0.1, 0.5, 0.8]

    def test_selection_init(self):
        selection = RouletteSelection(0.5)
        self.assertIsInstance(selection, RouletteSelection)

    def test_selection_unevaluated(self):
        for population_fraction in self.population_fractions:
            selection = RouletteSelection(population_fraction)
            selected_population = selection.__select__(self.test_population)
            
            self.assertNotEqual(
                selected_population.population_size,
                self.test_population.population_size
                )

            self.assertAlmostEqual(
                selected_population.population_size/self.test_population.population_size,
                population_fraction,
                places=1
                )

    def test_selection_evaluated(self):
        
        for max_fitness in self.max_fitness_values:
            
            for population_fraction in self.population_fractions:

                self.test_population = Population(1000, (100, 1))
                
                evaluated_individuals = self.test_population.individuals 

                evaluated_individuals[:evaluated_individuals.shape[0]//5]['fitness'] = max_fitness

                self.test_population.replace_individuals(evaluated_individuals)

                selection = RouletteSelection(population_fraction)

                selected_population = selection.__select__(self.test_population)

                self.assertNotEqual(
                    selected_population.population_size,
                    self.test_population.population_size
                    )

                self.assertEqual(
                    selected_population.max_fitness,
                    max_fitness
                    )

                self.assertLessEqual(
                    selected_population.population_size/self.test_population.population_size,
                    population_fraction
                    )

import unittest
from pystrand import Selection, Population, RandomSelection

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
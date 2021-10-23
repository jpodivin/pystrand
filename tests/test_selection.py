import unittest
from pystrand.operators.selections import *
from pystrand.populations import BasePopulation
from pystrand.genotypes import Genotype

class Dummy_Selection_Test(unittest.TestCase):
    test_population = BasePopulation(100, (100, 1))

    def test_selection_init(self):
        selection = BaseSelection()
        self.assertIsInstance(selection, BaseSelection)

    def test_selection(self):
        selection = BaseSelection()

        selected_population = selection.select(self.test_population)

        selected_population = BasePopulation(
                self.test_population.population_size,
                self.test_population.genome_shapes,
                self.test_population.gene_values,
                seed_individuals = selected_population
                )
        self.assertEqual(
            selected_population.population_size,
            self.test_population.population_size
            )

class Random_Selection_Test(unittest.TestCase):

    test_population = BasePopulation(1000000, (100, 1))
    selection_probabilities = [0.1, 0.5, 0.9]

    def test_selection_init(self):
        selection = RandomSelection(0.5)
        self.assertIsInstance(selection, RandomSelection)

    def test_selection(self):
        for selection_probability in self.selection_probabilities:
            selection = RandomSelection(selection_probability)
            selected_population = selection.select(self.test_population)

            selected_population = BasePopulation(
                self.test_population.population_size,
                self.test_population.genome_shapes,
                self.test_population.gene_values,
                seed_individuals = selected_population
                )

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

    test_population = BasePopulation(1000, (100, 1))
    population_fractions = [0.1, 0.5, 0.9]
    max_fitness_values = [0.1, 0.5, 0.8]

    def test_selection_init(self):
        selection = RouletteSelection(0.5)
        self.assertIsInstance(selection, RouletteSelection)

    def test_selection_unevaluated(self):
        for population_fraction in self.population_fractions:
            selection = RouletteSelection(population_fraction)
            selected_population = selection.select(self.test_population)
            selected_population = BasePopulation(
                self.test_population.population_size,
                self.test_population.genome_shapes,
                self.test_population.gene_values,
                seed_individuals = selected_population
                )

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

                self.test_population = BasePopulation(1000, (100, 1))

                evaluated_individuals = self.test_population.individuals

                evaluated_individuals[:evaluated_individuals.shape[0]//5]['fitness'] = max_fitness

                self.test_population.replace_individuals(evaluated_individuals)

                selection = RouletteSelection(population_fraction)

                selected_population = selection.select(self.test_population)
                selected_population = BasePopulation(
                    self.test_population.population_size,
                    self.test_population.genome_shapes,
                    self.test_population.gene_values,
                    seed_individuals = selected_population
                    )

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

class Elitism_Selection_Test(unittest.TestCase):

    test_population = BasePopulation(1000, (100, 1))
    population_fractions = [0.1, 0.5, 0.9]
    max_fitness_values = [0.1, 0.5, 0.8]

    def test_selection_init(self):
        selection = ElitismSelection(0.5)
        self.assertIsInstance(selection, ElitismSelection)

    def test_selection_unevaluated(self):
        for population_fraction in self.population_fractions:
            selection = ElitismSelection(population_fraction)
            selected_population = selection.select(self.test_population)
            selected_population = BasePopulation(
                    self.test_population.population_size,
                    self.test_population.genome_shapes,
                    self.test_population.gene_values,
                    seed_individuals = selected_population
                    )

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

                self.test_population = BasePopulation(1000, (100, 1))

                evaluated_individuals = self.test_population.individuals

                evaluated_individuals[:evaluated_individuals.shape[0]//5]['fitness'] = max_fitness

                self.test_population.replace_individuals(evaluated_individuals)

                selection = ElitismSelection(population_fraction)

                selected_population = selection.select(self.test_population)
                selected_population = BasePopulation(
                    self.test_population.population_size,
                    self.test_population.genome_shapes,
                    self.test_population.gene_values,
                    seed_individuals = selected_population
                    )

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

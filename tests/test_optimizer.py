from pystrand.optimizers import BaseOptimizer
from pystrand.genotypes import Genotype
from pystrand.populations import BasePopulation
import unittest
import numpy as np

target_genotypes_small = [
        np.zeros((10),),
        np.array([i%2 for i in range(10)]),
        np.array([i+1%2 for i in range(10)]),
        np.array([i%3 for i in range(10)])]

target_genotypes_large = [
    np.resize(array, (100,)) for array in target_genotypes_small
    ]


class FitnessFn:
    """Simple fitness function.
    The elements of the genotype array serve as coefficients
    for 1d polynomial Pg, evaluated at x=1.1
    Our goal is to find coefficients allowing for Pg(x) = Pt(x).

    For the sake of simplicity, and since we defined fitness in range <0.0, 1.0>,
    the value evaluated P is clipped and transformed.
    """
    def __init__(self, target_genotype):
        self.x = 1.1
        self.target_polynomial_val = np.poly1d(target_genotype)(self.x)

    def __call__(self, individual):
        polynomial = np.poly1d(individual)
        result = abs(polynomial(self.x)-self.target_polynomial_val)

        if result > 1.0:
            return 0.0
        else:
            return 1.0 - result


class Optimizer_init_test(unittest.TestCase):

    def test_optimizer_init_small(self):
        """
        Optimizer init test. Only checks genotype preservation and instantiation
        """

        for target_genotype in target_genotypes_small:

            fitness_fn = FitnessFn(target_genotype)
            target_genotype = Genotype(
                target_genotype.shape,
                gene_vals=np.unique(target_genotype),
                default_genome=target_genotype)

            population = BasePopulation(
                pop_size = np.sum(target_genotype.shape)*10,
                genome_shapes = target_genotype.shape,
                gene_vals = target_genotype.gene_vals,
                random_init = True)

            new_optimizer = BaseOptimizer(
                population,
                fitness_function=fitness_fn,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            self.assertIsInstance(new_optimizer, BaseOptimizer)

            self.assertEqual(new_optimizer._fitness_function, fitness_fn)

    def test_optimizer_init_large(self):
        """
        Optimizer init test. Only checks genotype preservation and instantiation
        """

        for target_genotype in target_genotypes_large:

            fitness_fn = FitnessFn(target_genotype)
            target_genotype = Genotype(
                target_genotype.shape,
                gene_vals=np.unique(target_genotype),
                default_genome=target_genotype)

            population = BasePopulation(
                pop_size = np.sum(target_genotype.shape)*10,
                genome_shapes = target_genotype.shape,
                gene_vals = target_genotype.gene_vals,
                random_init = True)

            new_optimizer = BaseOptimizer(
                population,
                fitness_function=fitness_fn,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            self.assertIsInstance(new_optimizer, BaseOptimizer)

            self.assertEqual(new_optimizer._fitness_function, fitness_fn)


class Optimizer_Run_test_sequential(unittest.TestCase):

    test_runtime_short = 10
    test_runtime_long = 100
    history_dict_keys = [
        'iteration',
        'max_fitness',
        'min_fitness',
        'fitness_avg',
        'fitness_std']

    def test_optimizer_run_small(self):
        """
        Short run of basic optimizer with default params and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_small:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = np.sum(target_genotype.shape)*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys())
                and set(history.keys()).issubset(self.history_dict_keys)
                )

            self.assertLessEqual(max(history['iteration']), self.test_runtime_short)

    def test_optimizer_run_large(self):
        """
        Long run of basic optimizer with default params and binary genome.
        10000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """
        for target_genotype in target_genotypes_large:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = np.sum(target_genotype.shape)*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                population = population,
                max_iterations = self.test_runtime_long,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys())
                and set(history.keys()).issubset(self.history_dict_keys))

            self.assertLessEqual(max(history['iteration']), self.test_runtime_long)

    def test_optimizer_elitism_small(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_small[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = 'elitism',
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness']))

    def test_optimizer_elitism_large(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_large[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = 'elitism',
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness']))


    def test_optimizer_combined_small(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_small[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = ['elitism', 'roulette'],
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s " %(target_genotype, history['max_fitness']))


    def test_optimizer_combined_large(self):
        """Long run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_large[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = ['elitism', 'roulette'],
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness']))


class Optimizer_Run_test_parallel(unittest.TestCase):

    test_runtime_short = 10
    test_runtime_long = 100
    history_dict_keys = [
        'iteration',
        'max_fitness',
        'min_fitness',
        'fitness_avg',
        'fitness_std'
        ]

    def test_optimizer_run_small(self):
        """
        Short run of basic optimizer with default params and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_small:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = np.sum(target_genotype.shape)*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5,
                parallelize = True)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys())
                and set(history.keys()).issubset(self.history_dict_keys)
                )

            self.assertLessEqual(max(history['iteration']), self.test_runtime_short)

    def test_optimizer_run_large(self):
        """
        Long run of basic optimizer with default params and binary genome.
        10000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """
        for target_genotype in target_genotypes_large:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = np.sum(target_genotype.shape)*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5,
                parallelize = True)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            self.assertIsInstance(history, dict)

            self.assertTrue(
                set(self.history_dict_keys).issubset(history.keys())
                and set(history.keys()).issubset(self.history_dict_keys)
                )

            self.assertLessEqual(max(history['iteration']), self.test_runtime_long)

    def test_optimizer_elitism_small(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_small[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = 'elitism',
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5,
                parallelize = True)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness'])
                    )

    def test_optimizer_elitism_large(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_large[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = 'elitism',
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5,
                parallelize = True)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness'])
                    )

    def test_optimizer_combined_small(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_small[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True
                )

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = ['elitism', 'roulette'],
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5,
                parallelize = True)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s " %(target_genotype, history['max_fitness'])
                    )

    def test_optimizer_combined_large(self):
        """
        Short run of basic optimizer with elitism and binary genome.
        1000 generations should be enough to reach an optimal match.
        However this is still stochastic process so the test will check:
            - ticks of algorithm
            - consistency of genotypes
            - returned history of training
        """

        for target_genotype in target_genotypes_large[1:]:

            fitness_fn = FitnessFn(target_genotype)
            population = BasePopulation(
                pop_size = target_genotype.size*10,
                genome_shapes = target_genotype.shape,
                gene_vals = np.unique(target_genotype),
                random_init = True)

            new_optimizer = BaseOptimizer(
                max_iterations = self.test_runtime_short,
                selection_ops = ['elitism', 'roulette'],
                population = population,
                mutation_prob = 0.1,
                crossover_prob = 0.5,
                parallelize = True)

            history = new_optimizer.fit(fitness_fn, verbose=0)

            if len(history['max_fitness']) > 1:
                self.assertLessEqual(
                    0,
                    np.diff(history['max_fitness']).min(),
                    msg="\nTarget genotype: %s \nMax_fitness: %s" %(target_genotype, history['max_fitness'])
                    )

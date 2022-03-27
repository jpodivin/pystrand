"""Tests for polynomial model class.
"""
import numpy as np

import unittest

import pystrand.models.polymodels as models
import pystrand.optimizers as opt

class TestPolyModel(unittest.TestCase):
    gene_domains = [
        np.arange(-1, 1, 0.1),
        np.arange(-1, 1, 0.01),
        np.arange(-1, 1, 0.001),
        np.arange(-1, 1, 0.0001),
        np.arange(0, 60, 1),]

    dummy_data = {
        'linear': np.polynomial.Polynomial([1, 2.5])
    }

    def test_model_init(self):
        for domain in self.gene_domains:
            model = models.PowerPolyModel(domain)
            expected_pop_size = 1000
            self.assertIsInstance(model.optimizer, opt.BaseOptimizer)
            self.assertEqual(model.optimizer.population.population_size, expected_pop_size)
            self.assertEqual(len(model.optimizer.population.genome_shapes), expected_pop_size)
            self.assertEqual(
                model.optimizer.population.genome_shapes[0], (min(len(domain), 10),))

    def test_model_run_short(self):
        for domain in self.gene_domains:
            model = models.PowerPolyModel(domain, max_iterations=10)
            test_samples = domain[domain % 0.02 < 0.1]
            test_labels = self.dummy_data['linear'](test_samples)
            model.fit(test_samples, test_labels)

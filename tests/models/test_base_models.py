"""Tests for BaseModel class.
"""
import numpy as np

import unittest

import pystrand.models.base_models as models
import pystrand.optimizers as opt

class TestBaseModel(unittest.TestCase):
    gene_domains = [
        np.arange(-1, 1, 0.1),
        np.arange(-1, 1, 0.01),
        np.arange(-1, 1, 0.001),
        np.arange(-1, 1, 0.0001),
        np.arange(0, 60, 1),]

    def test_model_init(self):
        for domain in self.gene_domains:
            model = models.BaseModel(domain)
            expected_pop_size = 1000
            self.assertIsInstance(model.optimizer, opt.BaseOptimizer)
            self.assertEqual(model.optimizer.population.population_size, expected_pop_size)
            self.assertEqual(len(model.optimizer.population.genome_shapes), expected_pop_size)
            self.assertEqual(
                model.optimizer.population.genome_shapes[0], (min(len(domain), 10),))

    def test_model_fit_exception(self):
        model = models.BaseModel(self.gene_domains[0])

        self.assertRaises(NotImplementedError, model.fit, [1], [1])

    def test_model_predict_exception(self):
        model = models.BaseModel(self.gene_domains[0])

        self.assertRaises(NotImplementedError, model.predict, [1])

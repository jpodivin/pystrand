"""
"""

import numpy as np
from pystrand.models.base_models import BaseModel
from pystrand import fitnessfunctions as fn

class PowerPolyModel(BaseModel):
    """Model as a power series polynomial with coeficients equivalent to genes.
    """
    def __init__(self, gene_domain, population_size=None, inverted_fitness=True, **kwargs):

        super().__init__(gene_domain, population_size=population_size, **kwargs)
        self._fitness_fn = fn.DataFitnessFn(inverted=inverted_fitness)

    def _infer_pop_params(self, domain):
        params = {}
        params['pop_size'] = min(int(np.around(np.sum(np.abs(domain)))), 1000)
        params['genome_shapes'] = (min(len(domain), 10), )
        params['gene_vals'] = domain
        return params

    def fit(self, X, y, **kwargs):
        """
        """
        self._fitness_fn.data = X
        self._fitness_fn.labels = y
        history = self._optimizer.fit(
            self._fitness_fn, kwargs.get('verbose', 1))
        return history

    def predict(self, x):
        """
        """
        genotype = self.optimizer.population.retrieve_best()[0]['genotype']
        pol = np.polynomial.Polynomial(genotype)
        val = pol(x)
        return val

    @property
    def optimizer(self):
        return self._optimizer

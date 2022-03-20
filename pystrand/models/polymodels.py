"""Polynomial model classes
"""

import numpy as np
from pystrand.models.base_models import BaseModel
from pystrand import fitnessfunctions as fn

class PowerPolyModel(BaseModel):
    """Model as a power series polynomial with coeficients equivalent to genes.
    """
    def __init__(self, gene_domain, population_size=None,
                 inverted_fitness=True, crossover_prob=0.5, **kwargs):

        super().__init__(
            gene_domain,
            population_size=population_size,
            crossover_prob=crossover_prob, **kwargs)

        self._fitness_fn = fn.PowerPolyFitnessFn(inverted=inverted_fitness)

    def fit(self, X, y, **kwargs):
        """Fit polynomial genetic algorithm model

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        dict
            Dictionary of recorded model statistics over time.
        """
        self._fitness_fn.data = X
        self._fitness_fn.labels = y
        history = self._optimizer.fit(
            self._fitness_fn, kwargs.get('verbose', 1))
        return history

    def predict(self, x):
        """Evaluate vector 'x'

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        float
            Evaluation of the modelled polynomial.
        """
        genotype = self.solution['genotype']
        pol = np.polynomial.Polynomial(genotype)
        val = pol(x)
        return val

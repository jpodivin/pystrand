"""Model classes
"""
from pystrand.populations import BasePopulation
from pystrand.optimizers import BaseOptimizer

class BaseModel:

    def __init__(self, gene_domain, population_size=None, **kwargs):

        inferred_parameters = self._infer_pop_params(gene_domain)

        if population_size is None:
            population_size = inferred_parameters['pop_size']

        genome_shapes = kwargs.get('genome_shapes', inferred_parameters['genome_shapes'])
        gene_vals = kwargs.get('gene_vals', inferred_parameters['gene_vals'])
        parallelize = kwargs.get('parallelize', True)

        population = BasePopulation(
            population_size,
            genome_shapes=genome_shapes,
            gene_vals=gene_vals)

        self._optimizer = BaseOptimizer(
            population,
            max_iterations=kwargs.get('max_iterations'),
            parallelize=parallelize)

        self._fitness_fn = None

    def _infer_pop_params(self, domain):
        """Guess general model parameters using heuristic
        """
        raise NotImplementedError

    def fit(self, X, y=None, **kwargs):
        """
        """
        return self._optimizer.fit(self._fitness_fn, kwargs.get('verbose', 1))

    def predict(self, x):
        """
        """
        raise NotImplementedError

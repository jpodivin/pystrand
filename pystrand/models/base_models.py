"""Basic model classes
The most general models are defined here.
More specialized cases are to be placed in separate submodules.
"""
from pystrand.populations import BasePopulation
from pystrand.optimizers import BaseOptimizer

class BaseModel:
    """Basic genetic algorithm model.
    Defines API for derived model classes, but isn't inteded for actual
    use as the key methods are to be implemented in the subclasses.
    """
    def __init__(self, gene_domain, population_size=None, **kwargs):

        inferred_parameters = self._infer_pop_params(gene_domain)

        if population_size is None:
            population_size = inferred_parameters['pop_size']

        genome_shapes = kwargs.get('genome_shapes', inferred_parameters['genome_shapes'])
        gene_vals = kwargs.get('gene_vals', inferred_parameters['gene_vals'])
        kwargs['parallelize'] = kwargs.get('parallelize', True)
        max_iterations = kwargs.pop('max_iterations', 0)

        population = BasePopulation(
            population_size,
            genome_shapes=genome_shapes,
            gene_vals=gene_vals)

        self._optimizer = BaseOptimizer(
            population,
            max_iterations=max_iterations,
            **kwargs)

        self._fitness_fn = None

    def _infer_pop_params(self, domain):
        """Guess general model parameters using heuristic

        Parameters
        ----------
        domain : np.ndarraytest_samples

        Returns
        -------
        dict
            Dictionary of inferred model parameters.
        """
        params = {
            'pop_size': 100,
            'genome_shapes': (min(len(domain), 10),),
            'gene_vals': domain
        }

        return params

    def fit(self, X, y=None, **kwargs):
        """Fit genetic algorithm model

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        """
        return self._optimizer.fit(self._fitness_fn, kwargs.get('verbose', 1))

    def predict(self, x):
        """Evaluate vector 'x'

        Parameters
        ----------
        x : np.ndarray
        """
        raise NotImplementedError

"""Basic model classes
The most general mode API is defined here.
More specialized cases are placed in separate submodules.
"""
from pystrand.populations import BasePopulation
from pystrand.optimizers import BaseOptimizer

class BaseModel:
    """Basic genetic algorithm model.
    Defines general API for models, but isn't inteded for actual
    use as the key methods are left to be implemented in the subclasses.
    """
    def __init__(self, gene_domain, population_size=None, **kwargs):

        inferred_parameters = self._infer_pop_params(gene_domain)

        if population_size is None:
            population_size = inferred_parameters['pop_size']

        genome_shapes = kwargs.get('genome_shapes', inferred_parameters['genome_shapes'])
        gene_vals = kwargs.get('gene_vals', inferred_parameters['gene_vals'])
        kwargs['parallelize'] = kwargs.get('parallelize', True)
        max_iterations = kwargs.pop('max_iterations', -1)

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
            'pop_size': 1000,
            'genome_shapes': (min(len(domain), 10),),
            'gene_vals': domain
        }

        return params

    def fit(self, X, y, **kwargs):
        """Fit genetic algorithm model

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        """
        raise NotImplementedError()

    def predict(self, x):
        """Evaluate vector 'x'

        Parameters
        ----------
        x : np.ndarray
        """
        raise NotImplementedError()

    @property
    def optimizer(self):
        """Return model optimizer.
        """
        return self._optimizer

    @property
    def solution(self):
        """Return best performing candidate solution.
        """
        return self._optimizer.population.retrieve_best()[0]

    @property
    def get_population(self):
        """Return entire population of candidate solutions.
        """
        return self._optimizer.population

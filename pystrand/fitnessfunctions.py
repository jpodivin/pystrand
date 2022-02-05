"""Fitness function and metrics for use by the Optimizer classes.
"""
import numpy as np

class MSELoss:
    """Simple MSE function.
    """
    def __call__(self, y, yprime):
        return np.square(np.subtract(yprime, y)).mean()

class MAXLoss:
    """Simple Max function.
    """
    def __call__(self, y, yprime):
        return np.abs(np.subtract(yprime, y)).max()

METRICS = {
    'mse': MSELoss(),
    'max': MAXLoss(),
}

class BaseFunction:
    """Base class of test functions.
    """
    def __init__(self, inverted=False):
        self._evaluated = 0
        self.inverted = inverted

    def __call__(self, values):
        """Evaluate function and increment evaluation counter.
        Interface is common to all subclasses of `BaseFunction`,
        the heavy lifting is performed by the __evaluate__ method.

        Parameters
        ----------
        values : np.ndarray
            Genotype submitted for evaluation

        Returns
        -------
        float
            Range depends on the specifics of the function,
            but most usually corresponds to <0,1>
        """
        evaluation = self.__evaluate__(values)
        self._evaluated += 1

        if self.inverted:
            evaluation = 1 / (1 + evaluation)

        return evaluation

    def __evaluate__(self, values):
        """Evaluate the function at a given point.
        Return results.

        Parameters
        ----------
        values : np.ndarray
            Genotype submitted for evaluation

        Returns
        -------
        float
            Range depends on the specifics of the function,
            but most usually corresponds to <0,1>
        """
        return 0.0

    def _optima(self, values):
        """Checks that provided values are among the known optimal points
        (minima or maxima depending on the function and task).
        Implementation of this method depends on properties
        of the tested function.

        Parameters
        ----------
        values : np.ndarray
            Genotype submitted for evaluation

        Returns
        -------
        bool
        """
        return False

    @property
    def evaluated(self):
        """Return how many times was the function evaluated.
        """
        return self._evaluated

    def optimum_reached(self, values):
        """Check if provided values represent one of the
        known optimal values.

        Parameters
        ----------
        values : np.ndarray
            Genotype submitted for evaluation

        Returns
        -------
        bool
        """
        return self._optima(values)


class SquashedDimsFunction(BaseFunction):
    """
    """
    def __init__(self, inverted, final_dimension, strategy='splitsum'):
        self._final_dimension = final_dimension
        self._squash_strategy = strategy
        super().__init__(inverted=inverted)

    def __call__(self, values):
        if self._squash_strategy == 'splitsum':
            values = np.reshape(values, (self._final_dimension, -1))
            values = np.sum(values, 1)
        return super().__call__(values)


class DataFitnessFn(BaseFunction):
    """Measure fitness of with against supplied samples and labels.
    """

    def __init__(self, inverted=False, metric='mse', **kwargs):

        self._metric = METRICS[metric]
        self.data = []
        self.labels = []
        super().__init__(inverted=inverted, **kwargs)

    def __evaluate__(self, values):

        phenotype = self._get_phenotype(values)
        predictions = [phenotype(sample) for sample in self.data]

        fitness = self._metric(predictions, self.labels)

        return fitness

    def _get_phenotype(self, genotype):
        """Return phenotype representation of given genotype.

        Parameters
        ----------
        genotype : np.ndarray
            Genotype submitted for evaluation

        Returns
        -------
        Evaluable phenotype
        """

        raise NotImplementedError()


class PowerPolyFitnessFn(DataFitnessFn):
    """Measure fitness of genome evaluated as list of coeficients for
    a power series polynomial.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _get_phenotype(self, genotype):
        phenotype = np.polynomial.Polynomial(genotype)
        return phenotype

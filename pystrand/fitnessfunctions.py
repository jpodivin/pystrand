"""
"""
import numpy as np

class MSELoss:
    def __call__(self, y, yprime):
        return np.square(np.subtract(yprime, y)).mean()

METRICS = {
    'mse': MSELoss()
}

class BaseFunction:
    """Base class of test functions.
    """
    def __init__(self, inverted=False):
        self._evaluated = 0
        self.inverted = inverted

    def __call__(self, values):
        """Evaluate function and increment evaluation counter.
        """
        evaluation = self.__evaluate__(values)
        self._evaluated += 1
        if self.inverted:
            evaluation = 1 / (1 + evaluation)

        return evaluation

    def __evaluate__(self, values):
        """Evaluate the function at a given point.
        Return results.
        :param values: function input
        :type values: numpy array

        :rtype: float
        """
        return 0.0

    def _optima(self, values):
        """Checks that provided values are among the known optimal points
        (minima or maxima depending on the function and task).
        Implementation of this method depends on properties
        of the tested function.

        :rtype: bool
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

        :param values: function input
        :type values: numpy array

        :rtype: bool
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
    """
    """

    def __init__(self, inverted=False, metric='mse'):

        self._metric = METRICS[metric]
        self.data = None
        self.labels = None
        super().__init__(inverted=inverted)

    def __evaluate__(self, values):

        phenotype = np.polynomial.Polynomial(values)
        predictions = [phenotype(sample) for sample in self.data]

        fitness = self._metric(predictions, self.labels)

        return fitness

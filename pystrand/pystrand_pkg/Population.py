import numpy as np

class Population(object):
    """Collection of individuals. """
    _individuals = np.array([])

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def populationSize(self):
        return self._individuals.size


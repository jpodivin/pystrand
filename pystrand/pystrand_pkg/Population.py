import numpy as np

class Population(object):
    """
    Collection of individual genomes.
    Provides facilities for working with multiple genomes at the same time.
    """
    _individuals = np.array([])

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def populationSize(self):
        return self._individuals.size

    @property
    def individuals(self):
        return self._individuals

import numpy as np

class BaseMutation:
    """
    Defines base mutation operator.
    Returning genotype unchanged.

    Parameters
    ----------

    None
    """
    def __init__(self):
        self._random_generator = np.random.default_rng()

    def __mutate__(self, genotype):
        return genotype

    def __call__(self, genotype):
        return self.__mutate__(genotype)

class PointMutation(BaseMutation):
    """
    Defines point mutation operator. Subclasses the BaseMutation.
    Changing at most one element of the genotype, with given probability.

    Parameters
    ----------

    probability : float
        Probability of changing random element of genotype.
        Default is 0.0
    """
    def __init__(self, probability=0.0):
        self._mutation_probability = probability
        super(PointMutation, self).__init__()

    def __mutate__(self, genotype):
        if genotype.size != 0:
            if self._random_generator.random() < self._mutation_probability:
                position = self._random_generator.choice(genotype.size)
                gene_vals_subset = np.setdiff1d(genotype._gene_vals, [genotype.flat[position]])
                genotype.flat[position] = self._random_generator.choice(gene_vals_subset)
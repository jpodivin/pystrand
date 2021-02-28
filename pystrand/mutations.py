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
        """
        Set up random generator to be used by mutation operator.
        """
        self._random_generator = np.random.default_rng()

    def __mutate__(self, genotype):
        """
        Apply mutation operator, in this case identity, on the given genotype.
        """
        pass

    def __call__(self, genotype):
        """
        Pass genotype to the __mutate__ method.
        """
        self.__mutate__(genotype)

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
        """
        Set probability of a point mutation.
        """
        self._mutation_probability = probability
        super(PointMutation, self).__init__()

    def __mutate__(self, genotype):
        """
        Apply mutation operator on the given genotype.
        The genotype mutates with probiblity given during initialization.

        New element (symbol/gene) value
        -------------------------------

        A single element (symbol or gene) of genotype, changes its value
        to one of the other values defined in the _gene_vals of given genotype.
        It is not possible for gene to remain the same.
        """
        if genotype.size != 0:
            if self._random_generator.random() < self._mutation_probability:
                position = self._random_generator.choice(genotype.size)
                gene_vals_subset = np.setdiff1d(genotype._gene_vals, [genotype.flat[position]])
                genotype.flat[position] = self._random_generator.choice(gene_vals_subset)
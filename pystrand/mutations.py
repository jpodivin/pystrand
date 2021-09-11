"""Mutation operators
"""
import numpy as np

class BaseMutation:
    """Defines base mutation operator.
    Returning genotype unchanged.

    Parameters
    ----------
    probability : float
        Probability of changing random element of genotype.
        Default is 0.0

    """
    def __init__(self, probability=0.0):
        """Set up random generator to be used by mutation operator.
        """
        self._random_generator = np.random.default_rng()
        self._mutation_probability = probability

    def __mutate__(self, genotype):
        """Apply mutation operator on the given genotype.
        """
        raise NotImplementedError()

    def _check_mutation(self, genotype):
        """Determine whether or not to apply mutation
        to the genotype.

        Parameters
        ----------
        genotype : Genotype
            Genotype has to have size > 0 and generated random number has to be
            greater than preset threshold.
            Conditions are located here to simplify development and troubleshooting.
        """
        return (
            genotype.size != 0 \
            and self._random_generator.random() > self._mutation_probability)

    def __call__(self, genotype):
        """Pass genotype to the __mutate__ method.

        Parameters
        ----------
        genotype : Genotype

        """
        if self._check_mutation(genotype):
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
    def __mutate__(self, genotype):
        """
        Apply mutation operator on the given genotype.
        The genotype mutates with probiblity given during initialization.

        New element (symbol/gene) value
        -------------------------------

        A single element (symbol or gene) of genotype, changes its value
        to one of the other values defined in the gene_vals of given genotype.
        It is not possible for gene to remain the same.
        """
        position = self._random_generator.choice(genotype.size)
        gene_vals_subset = np.setdiff1d(genotype.gene_vals, [genotype.flat[position]])
        genotype.flat[position] = self._random_generator.choice(gene_vals_subset)


class BlockMutation(BaseMutation):
    """Defines block mutation operator. Subclasses the BaseMutation.
    Changing several elements of the genotype at the same time, with given probability.

    Parameters
    ----------
    probability : float
        Probability of changing random block of genotype elements.
        Default is 0.0
    block_size : int
        Size of the mutation block.
        Default is 0, functionally equivalent to point mutation.

    """
    def __init__(self, probability=0.0, block_size=1):
        """
        Set probability of a block mutation and size of the block
        """
        self._block_size = block_size
        super(BlockMutation, self).__init__(probability)

    def __mutate__(self, genotype):

        position_counter = min(self._block_size, genotype.size)
        position = self._random_generator.choice(genotype.size)

        while position_counter > 0:
            genotype.flat[position] = self._random_generator.choice(genotype.gene_vals)
            position = (position+1)%genotype.size
            position_counter -= 1


class PermutationMutation(BaseMutation):
    """Defines Permutation mutation operator.
    Operator reorders elements of the genotype, either symbols
    or subarrays, along the given axis, with set probability.

    Parameters
    ----------
    probability : float
        Probability of genotype undergoing permutation.
        Default is 0.0
    axis : int
        Axis along which to permutate elements.
        Default is 0, or the first axis.

        Note
        ****
        Invalid values, for example axis >= number of tensor dimensions
        will result in exceptions or undefined behavior.

    Note
    ----
    Unlike other mutation operators, permutation can not introduce new
    symbols, or genes, into genotype. And has to be used in conjunction with
    other operators capable of doing so. In order to be useful.
    """
    def __init__(self, probability=0.0, axis=0):
        """Set probability for Permutation mutation
        and an axis along which to shuffle.
        """
        self._axis = axis
        super(PermutationMutation, self).__init__(probability)

    def __mutate__(self, genotype):
        """Changes order of genotype subarrays along the given axis.
        Uses numpy shuffle to obtain new permutation of the elements.
        """
        self._random_generator.shuffle(genotype, axis=self._axis)


class ShiftMutation(BaseMutation):
    """Defines Shift mutation operator.
    Flattened genotype array is shifted to the right by a chosen
    number of positions. Same number of leftmost positions of the
    flattened genotype is filled with symbols, randomly chosen
    from genotypes gene_vals.

    Parameters
    ----------
    probability : float
        Probability of changing random block of genotype elements.
        Default is 0.0
    shift_scale : int
        Number of places we want to shift right on the flattened genotype.
        Same number of symbols on the left side of the flattened genotype
        are replaced by randomly selected symbols from gene_vals.
    """
    def __init__(self, probability=0.0, shift_scale=1):
        self._shift_scale = shift_scale
        super(ShiftMutation, self).__init__(probability)

    def __mutate__(self, genotype):
        np.roll(genotype, self._shift_scale)
        genotype.flat[:self._shift_scale] = self._random_generator.choice(
            genotype.gene_vals,
            self._shift_scale)

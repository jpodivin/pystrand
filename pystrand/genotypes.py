import numpy as np
from pystrand.mutations import PointMutation

class Genotype(np.ndarray):
    """
    Genotype class, inherits from numpy ndarray and, in many ways,
    behaves like it.
    The code follows guidelines in: https://numpy.org/doc/stable/user/basics.subclassing.html
    """
    def __new__(
            cls,
            genome_shape,
            random_init=False,
            gene_vals=None,
            seed=0,
            default_genome=None,
            protected=False,
            **kwargs):
        """
        Return:
            New Genotype instance.
        """
        if random_init:
            random_generator = np.random.default_rng(seed=seed)
            genome = random_generator.choice(gene_vals, genome_shape)
        elif default_genome is not None:
            genome = default_genome
        else:
            genome = np.zeros(genome_shape)

        if gene_vals is None:
            gene_vals = [0, 1]

        genome = genome.view(cls)
        genome._gene_vals = gene_vals
        genome._protected = protected

        return genome

    def __array_finalize__(self, obj):
        """
        https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_finalize__
        """
        if obj is None:
            return

        self._genotype_fitness = getattr(obj, 'genotype_fitness', 0.0)
        self._gene_vals = getattr(obj, '_gene_vals', [0, 1])
        self._protected = getattr(obj, '_protected', False)

    def __reduce__(self):
        """
        Prepare object for pickling.
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__reduce__.html
        """
        pickled_genotype = super(Genotype, self).__reduce__()
        genotype_state = pickled_genotype[2] + (self._gene_vals, self._protected)

        return (pickled_genotype[0], pickled_genotype[1], genotype_state)

    def __setstate__(self, state):
        """
        Set value of attributes _gene_vals and _protected from state.
        """
        self._gene_vals = state[-2]
        self._protected = state[-1]

        super(Genotype, self).__setstate__(state[:-2])

    def mutate(self, mutation_op=PointMutation(0.01)):
        """
        Alters one gene (symbol) with given probability.
        New symbol is selected from subset of _gene_vals.

        Arguments:
            mutation_op -- Mutation operator, subtype of BaseMutation
        """
        mutation_op(self)

    def crossover(self, partner_genotype, mask=None):
        """
        Arguments:
            partner_genotype --
            mask -- determines which genes (symbols) are selected from parents.
                    If left as 'None' the mask is randomized each time.
                    Thus impacting performance.
        """
        if mask is None:
            #Random mask is used if none defined.
            mask = np.ndarray(self.genotype_shape, dtype=bool)

        descendant_genome = self.clone()
        descendant_genome[mask] = partner_genotype[mask]

        return descendant_genome

    def clone(self):
        """
        Returns copy of this Genome object.
        Genome, gene values and genome shape are all preserved.
        """

        return Genotype(
            self.genotype_shape,
            gene_vals=self.gene_vals,
            default_genome=self.copy())

    @property
    def genotype_shape(self):
        """
        Return shape of the genotype.

        Note
        ----

        Equivalent to the shape attribute of the numpy ndarray.
        """
        return self.shape

    @property
    def gene_vals(self):
        """
        Return _gene_vals.
        """
        return self._gene_vals

    @property
    def fitness(self):
        """
        Return _genotype_fitness attribute.
        """
        return self._genotype_fitness

    @property
    def protected(self):
        """
        Return _protected attribute.
        """
        return self._protected

    @protected.setter
    def protected(self, new_value):
        """
        Set _protected attribute/flag of the genotype.
        If _protected is `True` the genotype will not be altered by operators.
        """
        self._protected = new_value

    def set_fitness(self, new_fitness):
        """
        Set fitness of the genotype directly.
        Raises:
            TypeError: If 'new_fitness' isn't of type 'float'
        """
        if not isinstance(new_fitness, float):
            raise TypeError()

        self._genotype_fitness = new_fitness

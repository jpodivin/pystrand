import numpy as np

class Genotype(np.ndarray):
    """

    """
    def __new__(
            cls,
            genome_shape,
            random_init=False,
            gene_vals=[0, 1],
            seed=0,
            default_genome=None,
            protected=False,
            **kwargs):

        if random_init:
            genome = np.random.choice(gene_vals, genome_shape)
        elif default_genome is not None:
            genome = default_genome
        else:
            genome = np.zeros(genome_shape)

        genome = genome.view(cls)
        genome._gene_vals = gene_vals
        genome._protected = protected

        return genome

    def __array_finalized__(self, obj):

        if obj is None:
            return

        self._genotype_fitness = getattr(obj, 'genotype_fitness', 0.0)
        self._gene_vals = getattr(obj, '_gene_vals', [0, 1])
        self._protected = getattr(obj, '_protected', False)
    
    def __reduce__(self):

        pickled_genotype = super(Genotype, self).__reduce__()
        genotype_state = pickled_genotype[2] + (self._gene_vals, self._protected)

        return (pickled_genotype[0], pickled_genotype[1], genotype_state)

    def __setstate__(self, state):
        self._gene_vals = state[-2]
        self._protected = state[-1]

        super(Genotype, self).__setstate__(state[:-2])

    def mutate(self, mutation_prob=0.01):
        """
        Alters one gene (symbol) with given probability.
        New symbol is selected from subset of _gene_vals.

        Arguments:
            mutation_prob -- float in range [0, 1.0] inclusive.
                        Other values result in error, or undefined behavior.
        """
        if self.size != 0:
            if np.random.random_sample() < mutation_prob:
                position = np.random.choice(self.size)
                gene_vals_subset = np.setdiff1d(self._gene_vals, [self.flat[position]])
                self.flat[position] = np.random.choice(gene_vals_subset)

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
        return self.shape

    @property
    def gene_vals(self):
        return self._gene_vals

    @property
    def fitness(self):
        return self._genotype_fitness

    @property
    def protected(self):
        return self._protected

    @protected.setter
    def protected(self, new_value):
        self._protected = new_value

    def set_fitness(self, new_fitness):
        """
        Raises:
            TypeError: If 'new_fitness' isn't of type 'float'
        """
        if not isinstance(new_fitness, float):
            raise TypeError()

        self._genotype_fitness = new_fitness

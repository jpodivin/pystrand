import numpy as np

class Genotype:
	"""
	
	"""
	_genome = np.array([])
	_gene_vals = np.array([])
	_genotype_fitness = 0.0

	def __init__(self, 
			  genome_shape, 
			  random_init = False,
			  gene_vals = [0, 1],
			  seed = 0,
			  default_genome = None,
			  *args, 
			  **kwargs):

		self._gene_vals = gene_vals
		
		if random_init:
			self._genome = np.random.choice(self._gene_vals, genome_shape)
		elif default_genome is not None:
			self._genome = default_genome
		else:
			self._genome = np.zeros(genome_shape)

		super().__init__(*args, **kwargs)

	def mutate(self, mutation_prob=0.01):
		"""
		Alters one gene (symbol) with given probability.
		New symbol is selected from subset of _gene_vals.

		Arguments:
			mutation_prob -- float in range [0, 1.0] inclusive.
						 Other values result in error, or undefined behavior.
		"""
		if self._genome.size != 0:
			if np.random.random_sample() < mutation_prob:
				position = np.random.choice(self._genome.size)
				gene_vals_subset = np.setdiff1d(self._gene_vals, [self._genome.flat[position]])
				self._genome.flat[position] = np.random.choice(gene_vals_subset)

	def crossover(self, partner_genotype, mask = None):
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
			
		descendant_genome = np.copy(self._genome)
		descendant_genome[mask] = partner_genotype.genome[mask]	

		return Genotype(
			self.genotype_shape,
			gene_vals = self.gene_vals,
			default_genome = descendant_genome)

	def clone(self):
		"""
		Returns copy of this Genome object. 
		Genome, gene values and genome shape are all preserved. 
		"""
		
		return Genotype(
			self.genotype_shape, 
			gene_vals = self.gene_vals,
			default_genome = self.genome.copy())

	@property
	def genotype_shape(self):
		return self._genome.shape

	@property
	def genome(self):
		return self._genome

	@property
	def gene_vals(self):
		return self._gene_vals

	@property
	def fitness(self):
		return self._genotype_fitness

	def set_fitness(self, new_fitness):
		"""
		Raises:
			TypeError: If 'new_fitness' isn't of type 'float'
		"""
		if not isinstance(new_fitness, float):
			raise TypeError()

		self._genotype_fitness = new_fitness

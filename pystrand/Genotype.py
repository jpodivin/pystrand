import numpy as np

class Genotype:

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
		if self._genome.size != 0:
			if np.random.random_sample(1) < mutation_prob:
				position = np.random.choice(self._genome.size, 1)
				gene_vals_subset = (self._gene_vals != self._genome.flat[position]).flatten()
				self._genome.flat[position] = np.random.choice(
						self._gene_vals[gene_vals_subset], 
						1)

	def crossover(self, partner_genotype, mask):
		
		descendant_genome = np.copy(self._genome)
		descendant_genome[mask] = partner_genotype.genome[mask]	

		return Genotype(
			self.genotype_shape,
			gene_vals = self.gene_vals,
			default_genome = descendant_genome)

	def clone(self):
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
		if not isinstance(new_fitness, float):
			raise Exception()

		self._genotype_fitness = new_fitness
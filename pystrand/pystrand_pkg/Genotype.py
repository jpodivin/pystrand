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

		return super().__init__(*args, **kwargs)

	def mutate(self, mutation_prob=0.01):
		if self._genome.size != 0:
			if np.random.random(1) < mutation_prob:
				self._genome.flat[np.random.choice(
					self._genome.size, 1)] = np.random.choice(
						self._gene_vals, 1)

	def crossover(self, partner_genotypes, mask):
		for partner_genotypes in partner_genotypes:
			self._gene_vals[mask] = partner_genotypes.genome[mask]	

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
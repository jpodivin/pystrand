import numpy as np

class Genome:

	_genotype = np.array([])
	_gene_vals = np.array([])

	def __init__(self, 
			  genome_shape, 
			  random_init = False,
			  gene_vals = [0, 1],
			  seed = 0,
			  *args, 
			  **kwargs):

		self._gene_vals = gene_vals
		
		if random_init:
			self._genotype = np.random.choice(self._gene_vals, genome_shape)
		else:
			self._genotype = np.zeros(genome_shape)

		return super().__init__(*args, **kwargs)

	def mutate(self, mutation_prob=0.01):
		if self._genotype.size != 0:
			if np.random.random(1) < mutation_prob:
				self._genotype.flat[np.random.choice(self._genotype.size, 1)] = np.random.choice(self._gene_vals, 1)

	def crossover(self, partnerGenomes, pattern):
		pass

	def get_genotype_shape(self):
		return self._genotype.shape

	def get_genotype(self):
		return self._genotype

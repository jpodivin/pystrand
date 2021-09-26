# Low level API guide

## Populations

Population and optimizer are sufficient to construct a genetic algorithm by themselves,
or they can be encapsulated by a high level model API,
which allows for use of sklearn-like fit/predict API. 

Defining population entails setting population size,
allowed gene values, and shape of genotypes the algorithm will work on.

```
    population = BasePopulation(
        pop_size = 1000,
        genome_shapes = target_genotype.shape,
        gene_vals = np.unique(target_genotype))
```

## Fitness functions

In order to evaluate genotypes the optimizer requires a fitness function,
assigning a value in range [0, n] to every genotype in population.
Fitness function is essentially equivalent to loss function, as used in machine learning.

To be compatible with multiprocessing, the fitness function has to be a callable object.
The `BaseFunction` class takes care of call interface, letting us focus on how we want to define the actual fitness for the population.  

```
class FitnessFn(BaseFunction):

    def __init__(self, target_genotype):
        self.target_genotype = target_genotype
        super().__init__()

    def __evaluate__(self, individual):
        result = abs(individual - self.target_genotype)

        if result > 1.0:
            return 0.0
        return 1.0 - result

fitness_fn = FitnessFn(target_genotype)
```

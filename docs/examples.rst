Examples
--------

Suppose we seek a power polynomial function describing relationship between variables.
We can use genetic algorithm, or GA, to determine the coefficients of the function.

In our case the function is very simple `f(x) = 5 + 5x + 2x^2`.

.. code-block:: python

    >>>from pystrand.models.polymodels import PowerPolyModel
    >>>domain = [i/10 for i in range(-100, 100)]
    >>>x = [i for i in range(10)]
    >>>y = [5 + (5*i) + (2*(i**2)) for i in x]
    >>>model = PowerPolyModel(domain, population_size=500, max_iterations=1000, crossover_prob=0.5)
    >>>model.fit(x, y)

Setup of the GA instance consists of two steps,
definition of the genotype structure and genetic operator settings.

Genotype structure can be defined in terms of genotype size, shape,
and domain of values the individual genes can take. As such the genotype
can represent a string, a matrix or in our example coefficients of a function.

Various genetic operators can be applied to the individual genotypes,
with different probabilities and modes of operation.
Multiple operators of the same type can be applied sequentially, in various
orders, depending on the desired behavior of the algorithm. 

Both of these steps have profound impact on the algorithm performance.

Finally there is a question of the population size. Larger populations
can lead to algorithm covering larger portion of the search space initially.
However they can be very taxing on system resources.

Smaller populations can converge prematurely and in the extreme cases,
when population size approaches 1, the algorithm degenerates into hill climber heuristic.


.. note::
    Since genetic algorithms are fundamentally stochastic, the sought solution
    doesn't have to be reached after the same number of steps every time.
    On average, measured on 100 samples, the preceding example reached solution
    within 450 iterations.


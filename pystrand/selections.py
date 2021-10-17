"""Selection operators
"""
import numpy as np


class BaseSelection:
    """Base selection operator class.
    Doesn't apply any criteria and selects all individuals by default.
    """
    def __init__(
            self,
            **kwargs):
        self._name = kwargs.get("name", "Selection")
        self._rng = np.random.default_rng()

    def __select__(self, population):
        return population.individuals

    def select(self, population):
        """Public select method.

        Parameters
        ----------
        population : Population
            Population on which the operator will be applied.

        Returns
        -------
        np.ndarray : array of individuals
        """
        return np.array(
            self.__select__(population),
            dtype=population.individuals.dtype)


class RandomSelection(BaseSelection):
    """Randomly selects a fraction of individuals in given population.
    The selection probability is given as argument.
    """

    def __init__(
            self,
            selection_prob,
            *args,
            **kwargs):

        self._selection_prob = selection_prob

        super().__init__(*args, **kwargs)

    def __select__(self, population):

        selected_individuals = self._rng.choice(
            population.individuals,
            size=int(self._selection_prob*population.individuals.size))

        return selected_individuals


class RouletteSelection(BaseSelection):
    """
    Naive implementation of Roulette selection (or Fitness proportionate selection).
    Checks for case of maximum fitness = 0 and assignes equal probability to all individuals.

    Parameters
    ----------
    selected_population_fraction : float, required
    """

    def __init__(
            self,
            selected_population_fraction,
            *args,
            **kwargs):

        self._selected_population_fraction = selected_population_fraction

        super().__init__(*args, **kwargs)

    def __select__(self, population):
        n_selected = int(population.population_size*self._selected_population_fraction)
        probs = population.individuals["fitness"]
        if probs.min() < 0.0:
            probs = probs - probs.min()
        if probs.max() > 0.0:
            scaling = lambda x: x / np.sum(probs)
            probs = np.apply_along_axis(scaling, 0, probs)
        else:
            probs = np.full(probs.shape, 1.0/probs.size)

        selected_individuals = self._rng.choice(
            population.individuals,
            size=n_selected,
            p=probs)

        return selected_individuals


class ElitismSelection(BaseSelection):
    """Select n individuals with the highest fitness values.

    Parameters
    ----------
    selected_population_fraction : float, required
    """
    def __init__(
            self,
            selected_population_fraction,
            *args,
            **kwargs):

        self._selected_population_fraction = selected_population_fraction

        super().__init__(*args, **kwargs)

    def __select__(self, population):
        n_selected = int(population.population_size*self._selected_population_fraction)
        selected_individuals = population.retrieve_best(n_selected)

        for individual in selected_individuals['genotype']:
            individual.protected = True

        return selected_individuals

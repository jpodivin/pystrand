import multiprocessing as mp
import uuid

from pystrand.populations import BasePopulation
from pystrand.operators.selections import RouletteSelection, ElitismSelection, BaseSelection
from pystrand.operators.mutations import BaseMutation, PointMutation
from pystrand.loggers.csv_logger import CsvLogger
from pystrand.loggers.details import RunDetails


class BaseOptimizer:
    """Base optimizer class.

    Parameters
    ----------
    fitness_function : BaseFunction
        provides mapping from genotype to a fitness value, [0, 1]
    max_iterations : int
        0 by default
    population : Population
        Seed population, can include known sub-optimal solutions.
    mutation_prob : float
        0.001 by default
    mutation_ops :
        Mutation operator to use on genotypes.
        Uses supplied mutation_prob. If None, defaults to PointMutation.
        None by default.
    crossover_prob : float
        0.0 by default, no crossover will take place
    selection_ops :
    selected_fraction :
    log_path :
    parallelize : bool
        Use multiprocessing to evaluate genomes in parallel?

    Raises
    ------
    TypeError
        If supplied wrong selection method type.
        If supplied mutation_op not subclassing BaseMutation.

    """

    def __init__(self,
                 population,
                 max_iterations=0,
                 fitness_function=None,
                 mutation_prob=0.01,
                 mutation_ops=None,
                 crossover_prob=0.5,
                 selection_ops='roulette',
                 selected_fraction=0.1,
                 log_path=None,
                 parallelize=False,
                 **kwargs):
        """For each element in list of selection methods we check the type.
        Only Selection and string are accepted, other types raise TypeError.
        The strings must be reckognized as names of algorithm,
        any other string will result in ValueError.
        """
        self._optimizer_uuid = str(uuid.uuid1())
        self._fitness_function = fitness_function

        if mutation_ops:
            if isinstance(mutation_ops, list):
                self._mutation_ops = mutation_ops
            elif issubclass(type(mutation_ops), BaseMutation):
                self._mutation_ops = [mutation_ops]
            else:
                raise TypeError(
                    'Invalid mutation operator.',
                    type(mutation_ops))
        else:
            self._mutation_ops = [PointMutation(mutation_prob)]

        if log_path:
            self.logger = CsvLogger(log_path=log_path)

            if kwargs.get('save_details'):
                self.details_logger = RunDetails(log_path=log_path)
            else:
                self.details_logger = None
        else:
            self.logger = None
            self.details_logger = None

        self._crossover_probability = crossover_prob
        self._selection_methods = []
        self._parallelize = parallelize
        self._population = population
        self._max_iterations = max_iterations

        #First we turn selection_methods into list, in case it isn't.
        if not isinstance(selection_ops, list):
            selection_ops = [selection_ops]

        for selection_method in selection_ops:
            if isinstance(selection_method, str):
                if selection_method == 'roulette':
                    self._selection_methods += [RouletteSelection(selected_fraction)]
                elif selection_method == 'elitism':
                    self._selection_methods += [ElitismSelection(selected_fraction)]
                else:
                    raise ValueError(
                        'Unknown selection algorithm name.',
                        selection_method)
            elif isinstance(selection_method, BaseSelection):
                self._selection_methods += [selection_method]
            else:
                raise TypeError(
                    'Invalid selection type.',
                    type(selection_method))

    def evaluate_individual(self, individual):
        """Return fitness value of the given individual.
        """
        return self._fitness_function(individual)

    def evaluate_population(self):
        """Apply set fitness function to every individual in _population
        in either sequential or parallel manner depending on value of
        the _paralelize attribute. And store result in the 'fitness' field.
        """
        evaluated_individuals = self._population.individuals
        if self._parallelize:
            with mp.Pool() as worker_pool:
                result = worker_pool.map_async(
                    self._fitness_function,
                    evaluated_individuals['genotype']).get(5)
                evaluated_individuals['fitness'] = result
        else:
            evaluated_individuals['fitness'] = [
                self._fitness_function(individual)
                for individual
                in evaluated_individuals['genotype']]

        self._population.replace_individuals(evaluated_individuals)

    def select_genomes(self):
        """Create new population by sequentially applying selection operators
        in the order they were given to __init__.
        Expand the new population to match the original one.
        """
        new_population = BasePopulation(
            0,
            self._population.genome_shapes,
            self._population.gene_values)

        for selection_method in self._selection_methods:
            new_population.append_individuals(
                selection_method.select(self._population))

        new_population.expand_population(
            self._population.population_size)

        self._population = new_population

    def fit(self, fitnes_function=None, verbose=1):
        """Main training loop.
        Return statistics of the run as dictionary of lists.

        Parameters
        ----------
        fitness_function: BaseFunction

        verbose : int
            If not '0' outputs statistics using print every generation.
            Default is 1.

        """
        if fitnes_function:
            self._fitness_function = fitnes_function
        elif not self._fitness_function:
            raise RuntimeError("No fitness function supplied")

        run_id = uuid.uuid1()

        history = {
            "iteration" : [],
            "max_fitness" : [],
            "min_fitness" : [],
            "fitness_avg" : [],
            "fitness_std" : []}

        iteration = 0

        while iteration != self._max_iterations:
            try:
                self.evaluate_population()
            except mp.TimeoutError as timeoutException:
                print(
                    "Population evaluation timed out, with exception {}.".format(
                        timeoutException))
                break

            history["iteration"].append(iteration)
            history["max_fitness"].append(self._population.max_fitness)
            history["min_fitness"].append(self._population.min_fitness)
            history["fitness_avg"].append(self._population.avg_fitness)
            history["fitness_std"].append(self._population.fitness_std)

            if verbose > 0:
                print(" // ".join(
                    [key + ": " + str(record[-1]) for key, record in history.items()]
                    ))

            if self._population.max_fitness == 1.0:
                break

            self.select_genomes()

            self._population.mutate_genotypes(mutation_ops=self._mutation_ops)

            if self._crossover_probability > 0.0:
                self._population.cross_genomes(
                    crossover_prob=self._crossover_probability)

            iteration += 1

        if self.logger:
            self.logger.save_history(history, run_id=run_id)

        if self.details_logger:
            self.details_logger.save_run_details(self)

        return history

    @property
    def population(self):
        """Return optimized population.
        """
        return self._population

    @property
    def optimizer_uuid(self):
        """Return uuid of the optimizer.
        """
        return self._optimizer_uuid

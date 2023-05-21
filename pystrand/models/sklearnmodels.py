import numpy as np
from pystrand.populations import BasePopulation
from pystrand.optimizers import BaseOptimizer
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _num_features, check_random_state
from pystrand.fitnessfunctions import PowerPolyFitnessFn

class PowerPolyRegressor(BaseEstimator, RegressorMixin):
    """Model as a power series polynomial with coeficients equivalent to genes.
    """
    _required_parameters = ['gene_domain']
    def __init__(self, gene_domain, genome_shape = None,
                 population_size=100, crossover_prob=0.5,
                 random_state=None, loss='mse', **kwargs):

        self.gene_domain = gene_domain
        self.genome_shape = genome_shape
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        """Fit polynomial genetic algorithm model

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        PowerPolyRegressor
            Fitted regressor
        """
        check_X_y(X, y)

        self.n_features_in_ = _num_features(X)
        inferred_parameters = self._infer_pop_params()
        self.random_state_ = check_random_state(self.random_state)
        if self.genome_shape is None:
            self.genome_shape_ = inferred_parameters['genome_shape']
        else:
            self.genome_shape_ = self.genome_shape

        self.population_ = BasePopulation(
            self.population_size, self.genome_shape_,
            random_init=False,
            gene_vals=self.gene_domain)

        self.optimizer_ = BaseOptimizer(self.population_, random_state=self.random_state_)

        if not isinstance(self.loss, PowerPolyFitnessFn):
            self.loss_ = PowerPolyFitnessFn(metric=self.loss)
        else:
            self.loss_ = self.loss
        self.history_ = self.optimizer_.fit(
            self.loss_, kwargs.get('verbose', 1))
        self.is_fitted_ = True

        self.solution_ = self.population_.retrieve_best()[0]
        return self

    def predict(self, X):
        """Evaluate vectors in matrix 'X'

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        float
            Evaluation of the modeled polynomial.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        coeffs = self.solution_['genotype']
        y_pred = []
        for i in range(X.shape[0]):
            val = 0
            for j in range(X.shape[1]):
                
                pol = np.polynomial.Polynomial(coeffs[j])
                val += pol(X[i,j])
            y_pred.append(val)
        return np.array(y_pred)


    def _infer_pop_params(self):
        """Guess general model parameters using heuristic

        Parameters
        ----------
        domain : np.ndarraytest_samples

        Returns
        -------
        dict
            Dictionary of inferred model parameters.
        """
        params = {
            'pop_size': 1000,
            'genome_shape': (self.n_features_in_, min(len(self.gene_domain), 10),),
            'gene_vals': self.gene_domain
        }

        return params

    def _more_tags(self):
        return {
            "poor_score": True}
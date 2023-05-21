import unittest
from pystrand.models.sklearnmodels import PowerPolyRegressor
from sklearn.utils.estimator_checks import check_estimator

class TestSklearnRegressor(unittest.TestCase):

    def test_sklearn_check(self):
        check_estimator(PowerPolyRegressor([i for i in range(5)]))

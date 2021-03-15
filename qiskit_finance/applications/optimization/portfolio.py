from docplex.mp.advmodel import AdvModel
import numpy as np

from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class Portfolio(OptimizationApplication):

    def __init__(self, mu, sigma, risk_factor, budget):
        self._mu = mu
        self._sigma = sigma
        self._risk_factor = risk_factor
        self._budget = budget

    def to_quadratic_program(self):
        num_assets = len(self._mu)
        mdl = AdvModel(name='Portfolio')
        x = [mdl.binary_var(name='x_{0}'.format(i)) for i in range(num_assets)]
        quad = mdl.quad_matrix_sum(self._sigma, x)
        linear = np.dot(self._mu, x)
        mdl.minimize(self._risk_factor * quad + linear)
        mdl.add_constraint(mdl.sum(x[i] for i in range(num_assets)) == self._budget)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def portfolio_expected_value(self, result):
        return np.dot(self._mu, result.x)

    def portfolio_variance(self, result):
        return np.dot(result.x, np.dot(self._sigma, result.x))

    def interpret(self, result):
        x = self._result_to_x(result)
        return [i for i, x_i in enumerate(x) if x_i]

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma

    @property
    def risk_factor(self):
        return self._risk_factor

    @risk_factor.setter
    def risk_factor(self, risk_factor):
        self._risk_factor = risk_factor

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, budget):
        self._budget = budget

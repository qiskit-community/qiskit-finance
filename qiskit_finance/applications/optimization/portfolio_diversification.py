from docplex.mp.advmodel import AdvModel
import numpy as np

from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


class PortfolioDiversification(OptimizationApplication):

    def __init__(self, rho, num_assets, num_clusters):
        self._rho = rho
        self._num_assets = num_assets
        self._num_clusters = num_clusters

    def to_quadratic_program(self):
        mdl = AdvModel(name='Portfolio diversification')
        x = {(i, j): mdl.binary_var(name='x_{0}_{1}'.format(i, j)) for i in range(self._num_assets)
             for j in range(self._num_assets)}
        y = {i: mdl.binary_var(name='y_{0}'.format(i)) for i in range(self._num_assets)}
        mdl.maximize(mdl.sum(self._rho[i, j] * x[(i, j)]
                     for i in range(self._num_assets) for j in range(self._num_assets)))
        mdl.add_constraint(mdl.sum(y[j] for j in range(self._num_assets)) == self._num_clusters)
        for i in range(self._num_assets):
            mdl.add_constraint(mdl.sum(x[(i, j)] for j in range(self._num_assets)) == 1)
        for j in range(self._num_assets):
            mdl.add_constraint(x[(j, j)] == y[j])
        for i in range(self._num_assets):
            for j in range(self._num_assets):
                mdl.add_constraint(x[(i, j)] <= y[j])
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, result):
        x = self._result_to_x(result)
        return [i for i, x_i in enumerate(x[-self._num_assets:]) if x_i]

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho

    @property
    def num_assets(self):
        return self._num_assets

    @num_assets.setter
    def num_assets(self, num_assets):
        self._num_assets = num_assets

    @property
    def num_clusters(self):
        return self._num_clusters

    @num_clusters.setter
    def num_clusters(self, num_clusters):
        self._num_clusters = num_clusters



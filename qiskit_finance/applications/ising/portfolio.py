import copy

from docplex.mp.advmodel import AdvModel
import numpy as np

from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.applications.ising.base_application import BaseApplication


class Portfolio(BaseApplication):

    def __init__(self, mu, sigma, q, budget):
        self._mu = copy.deepcopy(mu)
        self._sigma = copy.deepcopy(sigma)
        self._q = copy.deepcopy(q)
        self._budget = copy.deepcopy(budget)

    def to_quadratic_program(self):
        num_assets = len(self._mu)
        mdl = AdvModel(name='portfolio')
        x = [mdl.binary_var(name='x_{0}'.format(i)) for i in range(num_assets)]
        quad = mdl.quad_matrix_sum(self._sigma, x)
        linear = np.dot(self._mu, x)
        mdl.minimize(quad + linear)
        mdl.add_constraint(mdl.sum(x[i] for i in range(num_assets)) == self._budget)
        qp = QuadraticProgram()
        qp.from_docplex(mdl)
        return qp

    def interpret(self, x):
        return x

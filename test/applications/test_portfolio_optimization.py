# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Portfolio Optimization class"""

import logging
import unittest
from test import QiskitFinanceTestCase
from ddt import ddt, data, unpack
import numpy as np

from qiskit.utils import algorithm_globals
from qiskit_optimization.problems import QuadraticProgram
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


@ddt
class TestPortfolioDiversification(QiskitFinanceTestCase):
    """Tests Portfolio Diversification application class."""

    def setUp(self):
        """Set up for the tests"""
        super().setUp()
        algorithm_globals.random_seed = 100
        self.num_assets = 4
        self.expected_returns = [0.01528439, -0.00078095, 0.00051792, 0.00087001]
        self.covariances = [
            [2.54138859e-03, 7.34022167e-05, 1.28600531e-04, -9.98612132e-05],
            [7.34022167e-05, 2.58486713e-04, 5.30427595e-05, 4.44816208e-05],
            [1.28600531e-04, 5.30427595e-05, 7.91504681e-04, -1.23887382e-04],
            [-9.98612132e-05, 4.44816208e-05, -1.23887382e-04, 1.97892585e-04],
        ]
        self.risk_factor = 0.5
        self.budget = 2
        self.bounds = [(0, 2), (0, 3), (0, 4), (1, 5)]

    def assertEqualQuadraticProgram(self, actual, expected):
        """Compare two instances for quadratic program"""
        # Test name
        self.assertEqual(actual.name, expected.name)
        # Test variables
        self.assertEqual(actual.get_num_vars(), expected.get_num_vars())
        for var1, var2 in zip(actual.variables, actual.variables):
            self.assertEqual(var1.vartype, var2.vartype)
            self.assertEqual(var1.lowerbound, var2.lowerbound)
            self.assertEqual(var1.upperbound, var2.upperbound)
        # Test objective
        self.assertEqual(actual.objective.sense, expected.objective.sense)
        self.assertEqual(actual.objective.constant, expected.objective.constant)
        self.assertDictEqual(actual.objective.linear.to_dict(), expected.objective.linear.to_dict())
        self.assertDictEqual(
            actual.objective.quadratic.to_dict(), expected.objective.quadratic.to_dict()
        )
        # Test constraint
        self.assertEqual(len(actual.linear_constraints), len(expected.linear_constraints))
        for act_lin, exp_lin in zip(actual.linear_constraints, expected.linear_constraints):
            self.assertEqual(act_lin.sense, exp_lin.sense)
            self.assertEqual(act_lin.rhs, exp_lin.rhs)
            self.assertEqual(act_lin.linear.to_dict(), exp_lin.linear.to_dict())

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        # When bounds is None
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget
        )
        actual_op = portfolio_optimization.to_quadratic_program()
        expected_op = QuadraticProgram(name="Portfolio optimization")
        for i in range(self.num_assets):
            expected_op.binary_var(name=f"x_{i}")
        quadratic = {
            (i, j): self.risk_factor * self.covariances[i][j]
            for i in range(self.num_assets)
            for j in range(self.num_assets)
        }
        linear = {i: -self.expected_returns[i] for i in range(self.num_assets)}
        expected_op.minimize(quadratic=quadratic, linear=linear)
        linear = {i: 1 for i in range(self.num_assets)}
        expected_op.linear_constraint(linear=linear, sense="==", rhs=self.budget)
        self.assertEqualQuadraticProgram(actual_op, expected_op)
        # When bounds is provided
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget, self.bounds
        )
        actual_op = portfolio_optimization.to_quadratic_program()
        expected_op = QuadraticProgram(name="Portfolio optimization")
        for i in range(self.num_assets):
            expected_op.integer_var(
                lowerbound=self.bounds[i][0], upperbound=self.bounds[i][1], name=f"x_{i}"
            )
        quadratic = {
            (i, j): self.risk_factor * self.covariances[i][j]
            for i in range(self.num_assets)
            for j in range(self.num_assets)
        }
        linear = {i: -self.expected_returns[i] for i in range(self.num_assets)}
        expected_op.minimize(quadratic=quadratic, linear=linear)
        linear = {i: 1 for i in range(self.num_assets)}
        expected_op.linear_constraint(linear=linear, sense="==", rhs=self.budget)
        self.assertEqualQuadraticProgram(actual_op, expected_op)

    @data(
        [[1], [[1], [1]], 0.5, 2, None],
        [[1], [[1, 1]], 0.5, 2, None],
        [[1, 2], [[1, 2], [3, 4]], 0.5, 2, [(0, 2), (3, 1)]],
        [[1, 2], [[1, 2], [3, 4]], 0.5, 2, [(0, 2), (0, 2), (0, 2)]],
    )
    @unpack
    def test_is_compatibility(self, expected_returns, covariances, risk_factor, budget, bounds):
        """Test error cases in _is_compatibility"""
        with self.assertRaises(QiskitFinanceError):
            _ = PortfolioOptimization(expected_returns, covariances, risk_factor, budget, bounds)

    def test_interpret(self):
        """test interpret"""
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget
        )
        result_x = np.array([0, 1, 0, 1])
        self.assertEqual(portfolio_optimization.interpret(result_x), [1, 3])

    def test_portfolio_expected_value(self):
        """test portfolio_expected_value"""
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget
        )
        result_x = np.array([0, 1, 0, 1])
        expected_value = np.dot(self.expected_returns, result_x)
        self.assertEqual(portfolio_optimization.portfolio_expected_value(result_x), expected_value)

    def test_portfolio_variance(self):
        """test portfolio_variance"""
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget
        )
        result_x = np.array([0, 1, 0, 1])
        variance = np.dot(result_x, np.dot(self.covariances, result_x))
        self.assertEqual(portfolio_optimization.portfolio_variance(result_x), variance)

    def test_risk_factor(self):
        """test risk factor"""
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget
        )
        portfolio_optimization.risk_factor = 0.898989
        self.assertEqual(portfolio_optimization.risk_factor, 0.898989)

    def test_budget(self):
        """test budget"""
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget
        )
        portfolio_optimization.budget = 3
        self.assertEqual(portfolio_optimization.budget, 3)

    def test_bounds(self):
        """test bounds"""
        portfolio_optimization = PortfolioOptimization(
            self.expected_returns, self.covariances, self.risk_factor, self.budget, self.bounds
        )
        portfolio_optimization.bounds = [(0, 4), (0, 4), (0, 4), (1, 4)]
        self.assertEqual(portfolio_optimization.bounds, [(0, 4), (0, 4), (0, 4), (1, 4)])


if __name__ == "__main__":
    unittest.main()

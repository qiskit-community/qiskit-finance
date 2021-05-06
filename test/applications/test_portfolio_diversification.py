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

""" Test Portfolio Diversification class"""

import logging
import unittest
from test import QiskitFinanceTestCase

import numpy as np
from qiskit.utils import algorithm_globals
from qiskit_optimization.problems import QuadraticProgram, VarType
from qiskit_finance.applications.optimization import PortfolioDiversification

logger = logging.getLogger(__name__)


class TestPortfolioDiversification(QiskitFinanceTestCase):
    """Tests Portfolio Diversification application class."""

    def setUp(self):
        """Set up for the tests"""
        super().setUp()
        algorithm_globals.random_seed = 100
        self.n = 2
        self.q = 1
        self.similarity_matrix = np.ones((self.n, self.n))
        self.similarity_matrix[0, 1] = 0.8
        self.similarity_matrix[1, 0] = 0.8

        # # self.instance = -1 * self.instance
        # self.qubit_op = get_operator(self.instance, self.n, self.q)

    def assertEqualQuadraticProgram(self, actual, expected):
        """Compare two instances for quadratic program"""
        # Test name
        self.assertEqual(actual.name, expected.name)
        # Test variables
        self.assertEqual(actual.get_num_vars(), expected.get_num_vars())
        for var in actual.variables:
            self.assertEqual(var.vartype, VarType.BINARY)
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

        # """ Compares the dags after unrolling to basis """
        # circuit_dag = circuit_to_dag(circuit)
        # expected_dag = circuit_to_dag(expected)

        # circuit_result = Unroller(basis).run(circuit_dag)
        # expected_result = Unroller(basis).run(expected_dag)

        # self.assertEqual(circuit_result, expected_result)

    def test_to_quadratic_program(self):
        """Test to_quadratic_program"""
        portfolio_diversification = PortfolioDiversification(
            similarity_matrix=self.similarity_matrix,
            num_assets=self.n,
            num_clusters=self.q,
        )

        actual_op = portfolio_diversification.to_quadratic_program()

        expected_op = QuadraticProgram(name="Portfolio diversification")
        for i in range(self.n):
            for j in range(self.n):
                expected_op.binary_var(name="x_{0}_{1}".format(i, j))
        for i in range(self.n):
            expected_op.binary_var(name="y_{0}".format(i))
        linear = {"x_0_0": 1, "x_0_1": 0.8, "x_1_0": 0.8, "x_1_1": 1}
        expected_op.maximize(linear=linear)
        expected_op.linear_constraint(linear={"y_0": 1, "y_1": 1}, sense="==", rhs=1)
        expected_op.linear_constraint(linear={"x_0_0": 1, "x_0_1": 1}, sense="==", rhs=1)
        expected_op.linear_constraint(linear={"x_1_0": 1, "x_1_1": 1}, sense="==", rhs=1)
        expected_op.linear_constraint(linear={"x_0_0": 1, "y_0": -1}, sense="==", rhs=0)
        expected_op.linear_constraint(linear={"x_1_1": 1, "y_1": -1}, sense="==", rhs=0)
        expected_op.linear_constraint(linear={"x_0_0": 1, "y_0": -1}, sense="<=", rhs=0)
        expected_op.linear_constraint(linear={"x_0_1": 1, "y_1": -1}, sense="<=", rhs=0)
        expected_op.linear_constraint(linear={"x_1_0": 1, "y_0": -1}, sense="<=", rhs=0)
        expected_op.linear_constraint(linear={"x_1_1": 1, "y_1": -1}, sense="<=", rhs=0)
        self.assertEqualQuadraticProgram(actual_op, expected_op)

    def test_interpret(self):
        """Test interpret"""
        portfolio_diversification = PortfolioDiversification(
            similarity_matrix=self.similarity_matrix,
            num_assets=self.n,
            num_clusters=self.q,
        )
        result_x = np.array([0, 1, 0, 1, 0, 1])
        self.assertEqual(portfolio_diversification.interpret(result_x), [1])

    def test_smilarity_matrix(self):
        """Test similarity_matrix"""
        portfolio_diversification = PortfolioDiversification(
            similarity_matrix=self.similarity_matrix,
            num_assets=self.n,
            num_clusters=self.q,
        )
        portfolio_diversification.similarity_matrix = np.array([[0, 1], [1, 0]])
        self.assertEqual(portfolio_diversification.similarity_matrix.tolist(), [[0, 1], [1, 0]])

    def test_num_assets(self):
        """test num_assets"""
        portfolio_diversification = PortfolioDiversification(
            similarity_matrix=self.similarity_matrix,
            num_assets=self.n,
            num_clusters=self.q,
        )
        portfolio_diversification.num_assets = 3
        self.assertEqual(portfolio_diversification.num_assets, 3)

    def test_num_clusters(self):
        """test num_clusters"""
        portfolio_diversification = PortfolioDiversification(
            similarity_matrix=self.similarity_matrix,
            num_assets=self.n,
            num_clusters=self.q,
        )
        portfolio_diversification.num_clusters = 3
        self.assertEqual(portfolio_diversification.num_clusters, 3)


if __name__ == "__main__":
    unittest.main()

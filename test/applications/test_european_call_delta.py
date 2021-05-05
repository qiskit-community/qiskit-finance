# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test European Call Expected Value uncertainty problem """

import unittest
from test import QiskitFinanceTestCase

from qiskit.utils import algorithm_globals
from qiskit.circuit.library import IntegerComparator
from qiskit.quantum_info import Operator
from qiskit_finance.applications.estimation import EuropeanCallDelta
from qiskit_finance.circuit.library.probability_distributions import UniformDistribution


class TestEuropeanCallDelta(QiskitFinanceTestCase):
    """Tests the EuropeanCallDelta application"""

    def setUp(self):
        super().setUp()
        self.seed = 457
        algorithm_globals.random_seed = self.seed

    def test_to_estimation_problem(self):
        """Test the expected circuit."""
        num_qubits = 3
        strike_price = 0.5
        bounds = (0, 2)
        # make an estimation problem
        uncertain_model = UniformDistribution(num_qubits)
        ecd = EuropeanCallDelta(num_qubits, strike_price, bounds, uncertain_model)
        est_problem = ecd.to_estimation_problem()
        # make a state_preparation circuit manually
        x = (strike_price - bounds[0]) / (bounds[1] - bounds[0]) * (2 ** num_qubits - 1)
        comparator = IntegerComparator(num_qubits, x)
        expected_circ = comparator.compose(uncertain_model, front=True)
        self.assertEqual(est_problem.objective_qubits, [num_qubits])
        self.assertTrue(Operator(est_problem.state_preparation).equiv(expected_circ))
        # pylint: disable=not-callable
        self.assertEqual(0.5, est_problem.post_processing(0.5))


if __name__ == "__main__":
    unittest.main()

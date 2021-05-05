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

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.utils import algorithm_globals
from qiskit_finance.applications.estimation import FixedIncomePricing
from qiskit_finance.circuit.library.probability_distributions import UniformDistribution
from qiskit_finance.circuit.library.payoff_functions import FixedIncomePricingObjective


class TestFixedIncomePricing(QiskitFinanceTestCase):
    """Tests the FixedIncomePricing application"""

    def setUp(self):
        super().setUp()
        self.seed = 457
        algorithm_globals.random_seed = self.seed

    def test_to_estimation_problem(self):
        """Test the expected circuit."""
        num_qubits = [2, 2]
        pca_matrix = np.eye(2)
        initial_interests = np.zeros(2)
        cash_flow = np.array([1, 2])
        rescaling_factor = 0.125
        bounds = [(0, 0.12), (0, 0.24)]
        # make an estimation problem
        uncertain_model = UniformDistribution(4)
        fip = FixedIncomePricing(
            num_qubits,
            pca_matrix,
            initial_interests,
            cash_flow,
            rescaling_factor,
            bounds,
            uncertain_model,
        )
        est_problem = fip.to_estimation_problem()
        # make a state_preparation circuit manually
        expected = QuantumCircuit(5)
        expected.cry(-np.pi / 216, 0, 4)
        expected.cry(-np.pi / 108, 1, 4)
        expected.cry(-np.pi / 27, 2, 4)
        expected.cry(-0.23271, 3, 4)
        expected.ry(9 * np.pi / 16, 4)
        expected_circ = expected.compose(uncertain_model, front=True)
        self.assertEqual(est_problem.objective_qubits, [4])
        self.assertTrue(Operator(est_problem.state_preparation).equiv(expected_circ))
        fipo = FixedIncomePricingObjective(
            num_qubits,
            pca_matrix,
            initial_interests,
            cash_flow,
            rescaling_factor,
            bounds,
        )
        # pylint: disable=not-callable
        self.assertEqual(fipo.post_processing(0.5), est_problem.post_processing(0.5))


if __name__ == "__main__":
    unittest.main()

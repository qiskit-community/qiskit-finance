# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test FixedIncomePricingObjective"""

import unittest
from test import QiskitFinanceTestCase

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.primitives import Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.circuit.library import NormalDistribution
from qiskit_finance.circuit.library.payoff_functions import FixedIncomePricingObjective


class TestFixedIncomePricingObjective(QiskitFinanceTestCase):
    """Tests FixedIncomePricingObjective"""

    def test_circuit(self):
        """Test the expected circuit."""
        num_qubits = [2, 2]
        pca = np.eye(2)
        initial_interests = np.zeros(2)
        cash_flow = np.array([1, 2])
        rescaling_factor = 0.125
        bounds = [(0, 0.12), (0, 0.24)]

        circuit = FixedIncomePricingObjective(
            num_qubits, pca, initial_interests, cash_flow, rescaling_factor, bounds
        )

        expected = QuantumCircuit(5)
        expected.cry(-np.pi / 216, 0, 4)
        expected.cry(-np.pi / 108, 1, 4)
        expected.cry(-np.pi / 27, 2, 4)
        expected.cry(-0.23271, 3, 4)
        expected.ry(9 * np.pi / 16, 4)

        self.assertTrue(Operator(circuit).equiv(expected))

    def test_application(self):
        """Test an end-to-end application."""

        a_n = np.eye(2)
        b = np.zeros(2)

        num_qubits = [2, 2]

        # specify the lower and upper bounds for the different dimension
        bounds = [(0, 0.12), (0, 0.24)]
        mu = [0.12, 0.24]
        sigma = 0.01 * np.eye(2)

        # construct corresponding distribution
        dist = NormalDistribution(num_qubits, mu, sigma, bounds=bounds)

        # specify cash flow
        c_f = [1.0, 2.0]

        # specify approximation factor
        rescaling_factor = 0.125

        # get fixed income circuit appfactory
        fixed_income = FixedIncomePricingObjective(
            num_qubits, a_n, b, c_f, rescaling_factor, bounds
        )

        # build state preparation operator
        state_preparation = fixed_income.compose(dist, front=True)

        problem = EstimationProblem(
            state_preparation=state_preparation,
            objective_qubits=[4],
            post_processing=fixed_income.post_processing,
        )

        sampler = Sampler(options={"shots": 1024, "seed": 12})
        iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, sampler=sampler)
        result = iae.estimate(problem)

        # compare to precomputed solution
        self.assertAlmostEqual(result.estimation_processed, 2.329154511815111)


if __name__ == "__main__":
    unittest.main()

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

from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.quantum_info import Operator
from qiskit.utils import algorithm_globals
from qiskit_finance.applications.estimation import EuropeanCallPricing
from qiskit_finance.circuit.library.probability_distributions import UniformDistribution


class TestEuropeanCallPricing(QiskitFinanceTestCase):
    """Tests the EuropeanCallPricing application"""

    def setUp(self):
        super().setUp()
        self.seed = 457
        algorithm_globals.random_seed = self.seed

    def test_to_estimation_problem(self):
        """Test the expected circuit."""
        num_qubits = 3
        rescaling_factor = 0.1
        strike_price = 0.5
        bounds = (0, 2)
        # make an estimation problem
        uncertain_model = UniformDistribution(num_qubits)
        ecp = EuropeanCallPricing(
            num_qubits, strike_price, rescaling_factor, bounds, uncertain_model
        )
        est_problem = ecp.to_estimation_problem()
        # make a state_preparation circuit manually
        breakpoints = [0, strike_price]
        slopes = [0, 1]
        offsets = [0, 0]
        image = (0, 2 - strike_price)
        domain = (0, 2)
        linear_function = LinearAmplitudeFunction(
            num_qubits,
            slopes,
            offsets,
            domain=domain,
            image=image,
            breakpoints=breakpoints,
            rescaling_factor=rescaling_factor,
        )
        expected_circ = linear_function.compose(uncertain_model, front=True)
        self.assertEqual(est_problem.objective_qubits, [num_qubits])
        self.assertTrue(Operator(est_problem.state_preparation).equiv(expected_circ))
        # pylint: disable=not-callable
        self.assertEqual(linear_function.post_processing(0.5), est_problem.post_processing(0.5))


if __name__ == "__main__":
    unittest.main()

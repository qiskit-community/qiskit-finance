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

""" Test EuropeanCallDelta."""

import unittest
from test import QiskitFinanceTestCase

import numpy as np

from qiskit.circuit.library import IntegerComparator
from qiskit.quantum_info import Operator
from qiskit.utils import QuantumInstance
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_finance.circuit.library import LogNormalDistribution

from qiskit_finance.circuit.library.payoff_functions import EuropeanCallDeltaObjective


class TestEuropeanCallDelta(QiskitFinanceTestCase):
    """Tests EuropeanCallDelta."""

    def test_circuit(self):
        """Test the expected circuit.

        If it equals the correct ``IntegerComparator`` we know the circuit is correct.
        """
        num_qubits = 3
        strike_price = 0.5
        bounds = (0, 2)
        ecd = EuropeanCallDeltaObjective(
            num_state_qubits=num_qubits, strike_price=strike_price, bounds=bounds
        )

        # map strike_price to a basis state
        x = (strike_price - bounds[0]) / (bounds[1] - bounds[0]) * (2 ** num_qubits - 1)
        comparator = IntegerComparator(num_qubits, x)

        self.assertTrue(Operator(ecd).equiv(comparator))

    def test_application(self):
        """Test an end-to-end application."""
        try:
            from qiskit import (
                Aer,
            )  # pylint: disable=unused-import,import-outside-toplevel
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        num_qubits = 3

        # parameters for considered random distribution
        s_p = 2.0  # initial spot price
        vol = 0.4  # volatility of 40%
        r = 0.05  # annual interest rate of 4%
        t_m = 40 / 365  # 40 days to maturity

        # resulting parameters for log-normal distribution
        mu = (r - 0.5 * vol ** 2) * t_m + np.log(s_p)
        sigma = vol * np.sqrt(t_m)
        mean = np.exp(mu + sigma ** 2 / 2)
        variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price;
        # in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - 3 * stddev)
        high = mean + 3 * stddev
        bounds = (low, high)

        # construct circuit factory for uncertainty model
        uncertainty_model = LogNormalDistribution(
            num_qubits, mu=mu, sigma=sigma ** 2, bounds=bounds
        )

        # set the strike price (should be within the low and the high value of the uncertainty)
        strike_price = 1.896

        # create amplitude function
        european_call_delta = EuropeanCallDeltaObjective(
            num_state_qubits=num_qubits, strike_price=strike_price, bounds=bounds
        )

        # create state preparation
        state_preparation = european_call_delta.compose(uncertainty_model, front=True)

        problem = EstimationProblem(
            state_preparation=state_preparation,
            objective_qubits=[num_qubits],
            post_processing=european_call_delta.post_processing,
        )

        # run amplitude estimation
        q_i = QuantumInstance(
            Aer.get_backend("aer_simulator"), seed_simulator=125, seed_transpiler=80
        )
        iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, quantum_instance=q_i)
        result = iae.estimate(problem)
        self.assertAlmostEqual(result.estimation_processed, 0.8079816552117238)


if __name__ == "__main__":
    unittest.main()

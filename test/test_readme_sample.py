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

"""
Code inside the test is the finance sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest

from test import QiskitFinanceTestCase


class TestReadmeSample(QiskitFinanceTestCase):
    """Test sample code from readme"""

    def test_readme_sample(self):
        """readme sample test"""
        # pylint: disable=import-outside-toplevel,redefined-builtin

        def print(*args):
            """overloads print to log values"""
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        import numpy as np
        from qiskit import BasicAer
        from qiskit.algorithms import AmplitudeEstimation
        from qiskit_finance.circuit.library import NormalDistribution
        from qiskit_finance.applications import FixedIncomePricing

        # Create a suitable multivariate distribution
        num_qubits = [2, 2]
        bounds = [(0, 0.12), (0, 0.24)]
        mvnd = NormalDistribution(
            num_qubits, mu=[0.12, 0.24], sigma=0.01 * np.eye(2), bounds=bounds
        )

        # Create fixed income component
        fixed_income = FixedIncomePricing(
            num_qubits,
            np.eye(2),
            np.zeros(2),
            cash_flow=[1.0, 2.0],
            rescaling_factor=0.125,
            bounds=bounds,
            uncertainty_model=mvnd,
        )

        # the FixedIncomeExpectedValue provides us with the necessary rescalings

        # create the A operator for amplitude estimation
        problem = fixed_income.to_estimation_problem()

        # Set number of evaluation qubits (samples)
        num_eval_qubits = 5

        # Construct and run amplitude estimation
        q_i = BasicAer.get_backend("statevector_simulator")
        algo = AmplitudeEstimation(num_eval_qubits=num_eval_qubits, quantum_instance=q_i)
        result = algo.estimate(problem)

        print("Estimated value:\t%.4f" % fixed_income.interpret(result))
        print("Probability:    \t%.4f" % result.max_probability)

        # ----------------------------------------------------------------------

        with self.subTest("test estimation"):
            self.assertAlmostEqual(result.estimation_processed, 2.46, places=4)
        with self.subTest("test max.probability"):
            self.assertAlmostEqual(result.max_probability, 0.8487, places=4)


if __name__ == "__main__":
    unittest.main()

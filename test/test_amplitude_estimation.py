# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Amplitude Estimation """

import unittest
from test import QiskitFinanceTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import (AmplitudeEstimation,
                               MaximumLikelihoodAmplitudeEstimation,
                               EstimationProblem)
from qiskit_finance.applications import (EuropeanCallDelta,
                                         FixedIncomeExpectedValue,
                                         EuropeanCallExpectedValue)


@ddt
class TestEuropeanCallOption(QiskitFinanceTestCase):
    """ Test European Call Option """

    def setUp(self):
        super().setUp()

        # number of qubits to represent the uncertainty
        num_uncertainty_qubits = 3

        # parameters for considered random distribution
        s_p = 2.0  # initial spot price
        vol = 0.4  # volatility of 40%
        r = 0.05  # annual interest rate of 4%
        t_m = 40 / 365  # 40 days to maturity

        # resulting parameters for log-normal distribution
        m_u = ((r - 0.5 * vol ** 2) * t_m + np.log(s_p))
        sigma = vol * np.sqrt(t_m)
        mean = np.exp(m_u + sigma ** 2 / 2)
        variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * m_u + sigma ** 2)
        stddev = np.sqrt(variance)

        # lowest and highest value considered for the spot price;
        # in between, an equidistant discretization is considered.
        low = np.maximum(0, mean - 3 * stddev)
        high = mean + 3 * stddev

        # set the strike price (should be within the low and the high value of the uncertainty)
        strike_price = 1.896

        # set the approximation scaling for the payoff function
        c_approx = 0.1

        # construct circuit factory for payoff function
        self.european_call = EuropeanCallExpectedValue(num_state_qubits=num_uncertainty_qubits,
                                                       strike_price=strike_price,
                                                       rescaling_factor=c_approx,
                                                       bounds=(low, high))

        # construct circuit factory for payoff function
        self.european_call_delta = EuropeanCallDelta(num_state_qubits=num_uncertainty_qubits,
                                                     strike_price=strike_price,
                                                     bounds=(low, high))

        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            seed_simulator=2,
                                            seed_transpiler=2)
        self._qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'), shots=100,
                                     seed_simulator=2, seed_transpiler=2)

    @idata([
        ['statevector', AmplitudeEstimation(3),
         {'estimation': 0.45868536404797905, 'mle': 0.1633160}],
        ['qasm', AmplitudeEstimation(4),
         {'estimation': 0.45868536404797905, 'mle': 0.23479973342434832}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 0.16330976193204114}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(3),
         {'estimation': 0.09784548904622023}],
    ])
    @unpack
    def test_expected_value(self, simulator, a_e, expect):
        """ expected value test """
        problem = EstimationProblem(state_preparation=self.european_call,
                                    objective_qubits=[3],
                                    post_processing=self.european_call.post_processing)
        # run simulation
        a_e.quantum_instance = self._qasm if simulator == 'qasm' else self._statevector
        result = a_e.estimate(problem)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(getattr(result, key), value, places=4,
                                   msg="estimate `{}` failed".format(key))

    @idata([
        ['statevector', AmplitudeEstimation(3),
         {'estimation': 0.8535534, 'mle': 0.8097974047170567}],
        ['qasm', AmplitudeEstimation(4),
         {'estimation': 0.8535534, 'mle': 0.8143597808556013}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 0.8097582003326866}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(6),
         {'estimation': 0.8096123776923358}],
    ])
    @unpack
    def test_delta(self, simulator, a_e, expect):
        """ delta test """
        problem = EstimationProblem(state_preparation=self.european_call_delta,
                                    objective_qubits=[3],
                                    post_processing=self.european_call_delta.post_processing)
        # run simulation
        a_e.quantum_instance = self._qasm if simulator == 'qasm' else self._statevector
        result = a_e.estimate(problem)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(getattr(result, key), value, places=4,
                                   msg="estimate `{}` failed".format(key))


@ddt
class TestFixedIncomeAssets(QiskitFinanceTestCase):
    """ Test Fixed Income Assets """

    def setUp(self):
        super().setUp()
        self._statevector = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                            seed_simulator=2,
                                            seed_transpiler=2)
        self._qasm = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'),
                                     shots=100,
                                     seed_simulator=2,
                                     seed_transpiler=2)

    @idata([
        ['statevector', AmplitudeEstimation(5),
         {'estimation': 2.4600, 'mle': 2.3402315559106843}],
        ['qasm', AmplitudeEstimation(5),
         {'estimation': 2.4600, 'mle': 2.3632087675061726}],
        ['statevector', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 2.340228883624973}],
        ['qasm', MaximumLikelihoodAmplitudeEstimation(5),
         {'estimation': 2.3174630932734077}]
    ])
    @unpack
    def test_expected_value(self, simulator, a_e, expect):
        """ expected value test """
        # can be used in case a principal component analysis
        # has been done to derive the uncertainty model, ignored in this example.
        a_n = np.eye(2)
        b = np.zeros(2)

        # get fixed income circuit
        fixed_income = FixedIncomeExpectedValue(num_qubits=[2, 2],
                                                pca_matrix=a_n,
                                                initial_interests=b,
                                                cash_flow=[1.0, 2.0],
                                                rescaling_factor=0.125,
                                                bounds=[(0., 0.12), (0., 0.24)])

        problem = EstimationProblem(state_preparation=fixed_income,
                                    objective_qubits=[4],
                                    post_processing=fixed_income.post_processing)
        # run simulation
        a_e.quantum_instance = self._qasm if simulator == 'qasm' else self._statevector
        result = a_e.estimate(problem)

        # compare to precomputed solution
        for key, value in expect.items():
            self.assertAlmostEqual(getattr(result, key), value, places=4,
                                   msg="estimate `{}` failed".format(key))


if __name__ == '__main__':
    unittest.main()

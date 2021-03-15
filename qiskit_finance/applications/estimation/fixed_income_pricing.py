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

"""The European Call Option Expected Value."""
from typing import Tuple, List

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.amplitude_estimators import EstimationProblem
from qiskit_finance.applications.estimation.estimation_application import EstimationApplication
from qiskit_finance.circuit.library.payoff_functions.fixed_income_pricing_objective \
    import FixedIncomePricingObjective


class FixedIncomePricing(EstimationApplication):
    """The European Call Option Delta.
    Evaluates the variance for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    def __init__(self,
                 num_qubits: List[int],
                 pca_matrix: np.ndarray,
                 initial_interests: List[int],
                 cash_flow: List[float],
                 rescaling_factor: float,
                 bounds: List[Tuple[float, float]],
                 uncertainty_model: QuantumCircuit
                 ) -> None:
        r"""
        Args:
            num_qubits: A list specifying the number of qubits used to discretize the assets.
            pca_matrix: The PCA matrix for the changes in the interest rates, :math:`\delta_r`.
            initial_interests: The initial interest rates / offsets for the interest rates.
            cash_flow: The cash flow time series.
            rescaling_factor: The scaling factor used in the Taylor approximation.
            bounds: The bounds for return values the assets can attain.
            uncertainty_model: A circuit for encoding a problem distribution
        """

        self._fixed_income = FixedIncomePricingObjective(
            num_qubits=num_qubits, pca_matrix=pca_matrix, initial_interests=initial_interests,
            cash_flow=cash_flow, rescaling_factor=rescaling_factor, bounds=bounds)
        self._state_preparation = self._fixed_income.compose(uncertainty_model, front=True)
        self._objective_qubits = uncertainty_model.num_qubits

    def to_estimation_problem(self):
        problem = EstimationProblem(state_preparation=self._state_preparation,
                                    objective_qubits=[self._objective_qubits],
                                    post_processing=self._fixed_income.post_processing)
        return problem

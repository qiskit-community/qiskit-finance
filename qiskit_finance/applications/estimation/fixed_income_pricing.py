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

"""An application class for the Fixed Income Pricing."""
from typing import Tuple, List

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.amplitude_estimators import (
    EstimationProblem,
    AmplitudeEstimatorResult,
)
from qiskit_finance.applications.estimation.estimation_application import (
    EstimationApplication,
)
from qiskit_finance.circuit.library.payoff_functions.fixed_income_pricing_objective import (
    FixedIncomePricingObjective,
)


class FixedIncomePricing(EstimationApplication):
    r"""An estimation application for the fixed income pricing problem.
    evaluate the expected value of the total value :math:`V` of the
    assets

    .. math::
        V = \sum_{t=1}^T \frac{c_t}{(1+r_t)^t}.

    [1]: Woerner, S., & Egger, D. J. (2018).
         Quantum Risk Analysis.
         `arXiv:1806.06893 <http://arxiv.org/abs/1806.06893>`_
    """

    def __init__(
        self,
        num_qubits: List[int],
        pca_matrix: np.ndarray,
        initial_interests: List[int],
        cash_flow: List[float],
        rescaling_factor: float,
        bounds: List[Tuple[float, float]],
        uncertainty_model: QuantumCircuit,
    ) -> None:
        r"""
        Args:
            num_qubits: A list specifying the number of qubits used to discretize the assets.
            pca_matrix: The PCA matrix for the changes in the interest rates, :math:`\delta_r`.
            initial_interests: The initial interest rates / offsets for the interest rates.
            cash_flow: The cash flow time series.
            rescaling_factor: The scaling factor used in the Taylor approximation.
            bounds: The list of the tuple of the bounds, (min, max), for return values the
                assets can attain.
            The bounds for return values the assets can attain.
            uncertainty_model: A circuit for encoding a problem distribution
        """

        self._objective = FixedIncomePricingObjective(
            num_qubits=num_qubits,
            pca_matrix=pca_matrix,
            initial_interests=initial_interests,
            cash_flow=cash_flow,
            rescaling_factor=rescaling_factor,
            bounds=bounds,
        )
        self._state_preparation = QuantumCircuit(self._objective.num_qubits)
        self._state_preparation.compose(
            uncertainty_model, range(uncertainty_model.num_qubits), inplace=True
        )
        self._state_preparation.compose(
            self._objective, range(self._objective.num_qubits), inplace=True
        )
        self._objective_qubits = uncertainty_model.num_qubits

    def to_estimation_problem(self) -> EstimationProblem:
        """Convert a problem instance into a
        `qiskit.algorithms.amplitude_estimators.EstimationProblem`

        Returns:
            The `qiskit.algorithms.amplitude_estimators.EstimationProblem` created
            from the Fixed problem instance.
        """
        problem = EstimationProblem(
            state_preparation=self._state_preparation,
            objective_qubits=[self._objective_qubits],
            post_processing=self._objective.post_processing,
        )
        return problem

    def interpret(self, result: AmplitudeEstimatorResult) -> float:
        """Convert the calculation result of the problem
        (`qiskit.algorithms.amplitude_estimators.AmplitudeEstimatorResult`)
        to the answer of the problem.

        Args:
            result: The calculated result of the problem

        Returns:
            The estimation value after the post_processing.
        """
        return result.estimation_processed

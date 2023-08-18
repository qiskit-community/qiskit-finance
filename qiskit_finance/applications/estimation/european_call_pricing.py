# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An application class for the European Call Pricing."""
from typing import Tuple

from qiskit.circuit import QuantumCircuit
from qiskit_algorithms import (
    EstimationProblem,
    AmplitudeEstimatorResult,
)
from qiskit_finance.applications.estimation.estimation_application import (
    EstimationApplication,
)
from qiskit_finance.circuit.library.payoff_functions.european_call_pricing_objective import (
    EuropeanCallPricingObjective,
)


class EuropeanCallPricing(EstimationApplication):
    """Estimation application for the European Call Option Expected Value.
    Evaluates the expected payoff for a European call option given an uncertainty model.
    The payoff function is f(S, K) = max(0, S - K) for a spot price S and strike price K.
    """

    def __init__(
        self,
        num_state_qubits: int,
        strike_price: float,
        rescaling_factor: float,
        bounds: Tuple[float, float],
        uncertainty_model: QuantumCircuit,
    ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits used to represent the random variable.
            strike_price: strike price of the European option
            rescaling_factor: approximation factor for linear payoff
            bounds: The tuple of the bounds, (min, max), of the discretized random variable.
            uncertainty_model: A circuit for encoding a problem distribution
        """
        self._objective = EuropeanCallPricingObjective(
            num_state_qubits=num_state_qubits,
            strike_price=strike_price,
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
        :class:`qiskit_algorithms.EstimationProblem`

        Returns:
            The :class:`qiskit_algorithms.EstimationProblem` created
            from the European call pricing problem instance.
        """
        problem = EstimationProblem(
            state_preparation=self._state_preparation,
            objective_qubits=[self._objective_qubits],
            post_processing=self._objective.post_processing,
        )
        return problem

    def interpret(self, result: AmplitudeEstimatorResult) -> float:
        """Convert the calculation result of the problem
        (:class:`qiskit_algorithms.AmplitudeEstimatorResult`)
        to the answer of the problem.

        Args:
            result: The calculated result of the problem

        Returns:
            The estimation value after the post_processing.
        """
        return result.estimation_processed

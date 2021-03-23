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

"""An application class for a portfolio optimization problem."""
from typing import List, Union

import numpy as np
from docplex.mp.advmodel import AdvModel

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.problems import QuadraticProgram


class PortfolioOptimization(OptimizationApplication):
    """Optimization application for the "portfolio optimization" [1] problem.

    References:
        [1]: "Portfolio optimization",
        https://en.wikipedia.org/wiki/Portfolio_optimization
    """

    def __init__(self, expected_returns: np.ndarray, covariances: np.ndarray,
                 risk_factor: float, budget: int) -> None:
        """
        Args:
            expected_returns: The expected returns for the assets.
            covariances: The covariances between the assets.
            risk_factor: The risk appetite of the decision maker.
            budget: The budget, i.e. the number of assets to be selected.
        """
        self._expected_returns = expected_returns
        self._covariances = covariances
        self._risk_factor = risk_factor
        self._budget = budget

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a portfolio optimization problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`.

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the portfolio optimization problem instance.
        """
        num_assets = len(self._expected_returns)
        mdl = AdvModel(name='Portfolio optimization')
        x = [mdl.binary_var(name='x_{0}'.format(i)) for i in range(num_assets)]
        quad = mdl.quad_matrix_sum(self._covariances, x)
        linear = np.dot(self._expected_returns, x)
        mdl.minimize(self._risk_factor * quad - linear)
        mdl.add_constraint(mdl.sum(x[i] for i in range(num_assets)) == self._budget)
        op = QuadraticProgram()
        op.from_docplex(mdl)
        return op

    def portfolio_expected_value(self, result: Union[OptimizationResult, np.ndarray]) -> float:
        """Returns the portfolio expected value based on the result.

        Args:
            result: The calculated result of the problem

        Returns:
            The portfolio expected value
        """
        x = self._result_to_x(result)
        return np.dot(self._expected_returns, x)

    def portfolio_variance(self, result: Union[OptimizationResult, np.ndarray]) -> float:
        """Returns the portfolio variance based on the result

        Args:
            result: The calculated result of the problem

        Returns:
            The portfolio variance
        """
        x = self._result_to_x(result)
        return np.dot(x, np.dot(self._covariances, x))

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as a list of asset indices

        Args:
            result: The calculated result of the problem

        Returns:
            The list of asset indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, x_i in enumerate(x) if x_i]

    @property
    def expected_returns(self) -> np.ndarray:
        """Getter of expected_returns

        Returns:
            The expected returns for the assets.
        """
        return self._expected_returns

    @expected_returns.setter
    def expected_returns(self, expected_returns: np.ndarray) -> None:
        """Setter of expected_returns

        Args:
            expected_returns: The expected returns for the assets.
        """
        self._expected_returns = expected_returns

    @property
    def covariances(self) -> np.ndarray:
        """Getter of covariances

        Returns:
            The covariances between the assets.
        """
        return self._covariances

    @covariances.setter
    def covariances(self, covariances: np.ndarray) -> None:
        """Setter of covariances

        Args:
            covariances: The covariances between the assets.
        """
        self._covariances = covariances

    @property
    def risk_factor(self) -> float:
        """Getter of risk_factor

        Returns:
            The risk appetite of the decision maker.
        """
        return self._risk_factor

    @risk_factor.setter
    def risk_factor(self, risk_factor: float) -> None:
        """Setter of risk_factor

        Args:
            risk_factor: The risk appetite of the decision maker.
        """
        self._risk_factor = risk_factor

    @property
    def budget(self) -> int:
        """Getter of budget

        Returns:
            The budget, i.e. the number of assets to be selected.
        """
        return self._budget

    @budget.setter
    def budget(self, budget: int) -> None:
        """Setter of budget

        Args:
            budget: The budget, i.e. the number of assets to be selected.
        """
        self._budget = budget

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

"""An application class for a portfolio diversification problem."""
from typing import List, Union

import numpy as np
from docplex.mp.advmodel import AdvModel

from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp


class PortfolioDiversification(OptimizationApplication):
    """Optimization application for the "portfolio diversification" problem introduced in [1].

    References:
        [1]: GG. Cornuejols and R. Tutuncu, Optimization methods in finance, 2006
    """

    def __init__(self, similarity_matrix: np.ndarray, num_assets: int, num_clusters: int) -> None:
        """
        Args:
            similarity_matrix: An asset-to-asset similarity matrix, such as the covariance matrix.
            num_assets: The number of assets.
            num_clusters: The number of clusters of assets to output.
        """
        self._similarity_matrix = similarity_matrix
        self._num_assets = num_assets
        self._num_clusters = num_clusters

    def to_quadratic_program(self) -> QuadraticProgram:
        """Convert a portfolio diversification problem instance into a
        :class:`~qiskit_optimization.problems.QuadraticProgram`.

        Returns:
            The :class:`~qiskit_optimization.problems.QuadraticProgram` created
            from the portfolio diversification problem instance.
        """
        mdl = AdvModel(name="Portfolio diversification")
        x = {
            (i, j): mdl.binary_var(name="x_{0}_{1}".format(i, j))
            for i in range(self._num_assets)
            for j in range(self._num_assets)
        }
        y = {i: mdl.binary_var(name="y_{0}".format(i)) for i in range(self._num_assets)}
        mdl.maximize(
            mdl.sum(
                self._similarity_matrix[i, j] * x[(i, j)]
                for i in range(self._num_assets)
                for j in range(self._num_assets)
            )
        )
        mdl.add_constraint(mdl.sum(y[j] for j in range(self._num_assets)) == self._num_clusters)
        for i in range(self._num_assets):
            mdl.add_constraint(mdl.sum(x[(i, j)] for j in range(self._num_assets)) == 1)
        for j in range(self._num_assets):
            mdl.add_constraint(x[(j, j)] == y[j])
        for i in range(self._num_assets):
            for j in range(self._num_assets):
                mdl.add_constraint(x[(i, j)] <= y[j])
        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: Union[OptimizationResult, np.ndarray]) -> List[int]:
        """Interpret a result as a list of asset indices

        Args:
            result: The calculated result of the problem

        Returns:
            The list of asset indices whose corresponding variable is 1
        """
        x = self._result_to_x(result)
        return [i for i, x_i in enumerate(x[-self._num_assets :]) if x_i]

    @property
    def similarity_matrix(self) -> np.ndarray:
        """Getter of similarity_matrix

        Returns:
            An asset-to-asset similarity matrix, such as the covariance matrix.
        """
        return self._similarity_matrix

    @similarity_matrix.setter
    def similarity_matrix(self, similarity_matrix: np.ndarray):
        """Setter of similarity_matrix

        Args:
            similarity_matrix: An asset-to-asset similarity matrix, such as the covariance matrix.
        """
        self._similarity_matrix = similarity_matrix

    @property
    def num_assets(self) -> int:
        """Getter of num_assets

        Returns:
            The number of assets.
        """
        return self._num_assets

    @num_assets.setter
    def num_assets(self, num_assets: int):
        """Setter of num_assets

        Args:
            num_assets: The number of assets.
        """
        self._num_assets = num_assets

    @property
    def num_clusters(self) -> int:
        """Getter of num_clusters

        Returns:
             The number of clusters of assets to output
        """
        return self._num_clusters

    @num_clusters.setter
    def num_clusters(self, num_clusters: int) -> None:
        """Setter of num_clusters

        Args:
            num_clusters: The number of clusters of assets to output
        """
        self._num_clusters = num_clusters

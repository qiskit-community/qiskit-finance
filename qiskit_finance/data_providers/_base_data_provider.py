# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module implements the abstract base class for data provider modules in the finance module.

The module defines the :code:`BaseDataProvider` abstract class which should be inherited by any data
provider class within the finance module. It also includes the :code:`StockMarket` :code:`Enum`
representing supported stock markets.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import cast
import logging
from enum import Enum

import numpy as np
import fastdtw

from qiskit_algorithms.utils import algorithm_globals
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


class StockMarket(Enum):
    """:code:`Enum` representing various stock markets.

    This :code:`Enum` contains identifiers for the following stock markets,
    represented by their respective codes:

        * :code:`"XLON"`: The London Stock Exchange.
        * :code:`"XPAR"`: The Euronext Paris.
        * :code:`"XSES"`: The Singapore Exchange.

    """

    LONDON: str = "XLON"
    EURONEXT: str = "XPAR"
    SINGAPORE: str = "XSES"


class BaseDataProvider(ABC):
    """The abstract base class for :code:`data_provider` modules within Qiskit Finance.

    Creates :code:`data_provider` module subclasses based on the :code:`BaseDataProvider`
    abstract class in this module.
    Doing so requires that the required driver interface is implemented.

    To use the subclasses, please see
    https://qiskit-community.github.io/qiskit-finance/tutorials/11_time_series.html
    """

    @abstractmethod
    def __init__(self) -> None:
        self._data: list | None = None
        self._n: int = 0  # pylint: disable=invalid-name
        self.period_return_mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.period_return_cov: np.ndarray | None = None
        self.rho: np.ndarray | None = None
        self.mean: np.ndarray | None = None

    @abstractmethod
    def run(self) -> None:
        """
        Abstract method to load data.

        Method responsible for loading data. Subclasses of :code:`BaseDataProvider`
        must implement this method to load data from a specific data source.
        """
        pass

    def _check_data_loaded(self) -> None:
        """
        Checks if data is loaded.

        Raises:
            QiskitFinanceError: If no data is loaded. Please run the method :code:`run()`
                first to load the data.
        """
        if not hasattr(self, "_data") or not self._data:
            raise QiskitFinanceError(
                "No data loaded yet. Please run the method `run()` first to load the data."
            )

    # it does not have to be overridden in non-abstract derived classes.
    def get_mean_vector(self) -> np.ndarray:
        """Returns the mean value vector of each asset.

        Calculates the mean value for each asset based on the loaded data,
        assuming each row represents a time-series observation for an asset.

        Returns:
            np.ndarray: A vector containing the mean value of each asset.

        Raises:
            QiskitFinanceError: If no data is loaded. Please run the method :code:`run()`
                first to load the data.
        """
        self._check_data_loaded()

        self.mean = cast(np.ndarray, np.mean(self._data, axis=1))
        return self.mean

    @staticmethod
    def _divide(val_1, val_2):
        if val_2 == 0:
            if val_1 == 0:
                return 1
            logger.warning("Division by 0 on values %f and %f", val_1, val_2)
            return np.nan
        return val_1 / val_2

    # it does not have to be overridden in non-abstract derived classes.
    def get_period_return_mean_vector(self) -> np.ndarray:
        """
        Calculates the mean period return vector for each asset.

        Returns the mean period return value for each asset based on the loaded data.
        Period return is calculated as the ratio of the current period's value to
        the previous period's value minus one.

        Returns:
            np.ndarray: A vector containing the mean period return value of each asset.

        Raises:
            QiskitFinanceError: If no data is loaded. Please run the method :code:`run()`
                first to load the data.
        """
        self._check_data_loaded()

        _div_func = np.vectorize(BaseDataProvider._divide)
        period_returns = _div_func(np.array(self._data)[:, 1:], np.array(self._data)[:, :-1]) - 1
        self.period_return_mean = cast(np.ndarray, np.mean(period_returns, axis=1))
        return self.period_return_mean

    # it does not have to be overridden in non-abstract derived classes.
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Calculates the covariance matrix of asset returns.

        Returns the covariance matrix of asset returns based on the loaded data.
        Each row in the data is assumed to represent a time-series observation for an asset.
        Covariance measures the relationship between two assets, indicating how they move in relation
        to each other.

        Returns:
            np.ndarray: An asset-to-asset covariance matrix.

        Raises:
            QiskitFinanceError: If no data is loaded. Please run the method :code:`run()`
                first to load the data.
        """
        self._check_data_loaded()

        self.cov = np.cov(self._data, rowvar=True)
        return self.cov

    # it does not have to be overridden in non-abstract derived classes.
    def get_period_return_covariance_matrix(self) -> np.ndarray:
        """
        Calculates the covariance matrix of period returns for each asset.

        Returns the covariance matrix of period returns for each asset based
        on the loaded data. Period return is calculated as the ratio of the
        current period's value to the previous period's value minus one.
        Covariance measures the relationship between two assets' period
        returns, indicating how they move in relation to each other.

        Returns:
            np.ndarray: A covariance matrix between period returns of assets.

        Raises:
            QiskitFinanceError: If no data is loaded. Please run the method :meth:`run()`
                first to load the data.
        """
        self._check_data_loaded()

        _div_func = np.vectorize(BaseDataProvider._divide)
        period_returns = _div_func(np.array(self._data)[:, 1:], np.array(self._data)[:, :-1]) - 1
        self.period_return_cov = np.cov(period_returns)
        return self.period_return_cov

    # it does not have to be overridden in non-abstract derived classes.
    def get_similarity_matrix(self) -> np.ndarray:
        """
        Calculates the similarity matrix based on time-series using dynamic
        time warping.

        Returns the similarity matrix based on time-series using the
        approximate Dynamic Time Warping (DTW) algorithm that provides
        optimal or near-optimal alignments with an :math:`O(N)` time and memory
        complexity. DTW is a technique to measure the
        similarity between two sequences that may vary in time or speed.
        The resulting similarity matrix indicates the similarity between
        different assets' time-series data.

        Returns:
            np.ndarray: An asset-to-asset similarity matrix.

        Raises:
            QiskitFinanceError: If no data is loaded. Please run the method :meth:`run()`
                first to load the data.
        """
        self._check_data_loaded()

        self.rho = np.zeros((self._n, self._n))
        for i_i in range(0, self._n):
            self.rho[i_i, i_i] = 1.0
            for j_j in range(i_i + 1, self._n):
                this_rho, _ = fastdtw.fastdtw(self._data[i_i], self._data[j_j])
                this_rho = 1.0 / this_rho
                self.rho[i_i, j_j] = this_rho
                self.rho[j_j, i_i] = this_rho
        return self.rho

    # gets coordinates suitable for plotting
    # it does not have to be overridden in non-abstract derived classes.
    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates random coordinates for visualization purposes.

        Returns random coordinates for visualization purposes. These coordinates
        can be used to plot assets in a two-dimensional space, facilitating visualization
        of relationships between assets.

        Returns:
            tuple[np.ndarray, np.ndarray]: :math:`x` and :math:`y` coordinates of each asset.

        Note:
            The generated coordinates are random and may not reflect any meaningful relationship
            between assets.
        """
        x_c = np.zeros([self._n, 1])
        y_c = np.zeros([self._n, 1])
        x_c = (algorithm_globals.random.random(self._n) - 0.5) * 1
        y_c = (algorithm_globals.random.random(self._n) - 0.5) * 1
        # for (cnt, s) in enumerate(self.tickers):
        # x_c[cnt, 1] = self.data[cnt][0]
        # y_c[cnt, 0] = self.data[cnt][-1]
        return x_c, y_c

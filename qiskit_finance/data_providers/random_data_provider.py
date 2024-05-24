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

""" Pseudo-randomly generated mock stock-market data provider """
from __future__ import annotations
import datetime
import logging
import numpy as np

from ._base_data_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class RandomDataProvider(BaseDataProvider):
    """Pseudo-randomly generated mock stock-market data provider."""

    def __init__(
        self,
        tickers: str | list[str] | None = None,
        start: datetime.datetime = datetime.datetime(2016, 1, 1),
        end: datetime.datetime = datetime.datetime(2016, 1, 30),
        seed: int | None = None,
    ) -> None:
        """
        Initialize RandomDataProvider.

        Args:
            tickers (str | list[str] | None): Tickers for the data provider.
                Default is None, using ["TICKER1", "TICKER2"] if not provided.
            start (datetime.datetime): Start date of the data.
                Defaults to January 1st, 2016.
            end (datetime.datetime): End date of the data.
                Defaults to January 30th, 2016.
            seed (int | None): Random seed for reproducibility.
        """
        super().__init__()
        tickers = tickers if tickers is not None else ["TICKER1", "TICKER2"]

        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace("\n", ";").split(";")

        self._n = len(self._tickers)
        self._start = start
        self._end = end
        self._seed = seed

    def run(self) -> None:
        """
        Generate pseudo-random stock market data.

        Generates pseudo-random stock market data using normal distribution
        and truncates values to zero after the first occurrence of zero.

        Raises:
            None
        Returns:
            None
        """
        length = (self._end - self._start).days
        generator = np.random.default_rng(self._seed)
        self._data = []

        for _ in self._tickers:
            random_numbers = generator.standard_normal(length)
            cumsum = np.cumsum(random_numbers)
            d_f = cumsum + generator.integers(1, 101)
            trimmed = np.maximum(d_f, np.zeros(length))

            # Set all values after the first 0 to 0
            for idx, val in enumerate(trimmed):
                if val == 0:
                    trimmed[idx + 1 :] = 0
                    break

            self._data.append(trimmed.tolist())

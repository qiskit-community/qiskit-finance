# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Yahoo data provider. """

from typing import Optional, Union, List
import datetime
import logging
import tempfile
import yfinance as yf

from ._base_data_provider import BaseDataProvider
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)

# Sets Y!Finance cache path in a new temp folder.
# This is done to avoid race conditions in the same cache file
# from different processes.
# The path will be automatically deleted if this module unloads cleanly.
# This needs to be done during yfinance initialization before any call
_temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
yf.set_tz_cache_location(_temp_dir.name)


class YahooDataProvider(BaseDataProvider):
    """Yahoo data provider.

    Please see:
    https://qiskit.org/ecosystem/finance/tutorials/11_time_series.html
    for instructions on use.
    """

    def __init__(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        start: datetime.datetime = datetime.datetime(2016, 1, 1),
        end: datetime.datetime = datetime.datetime(2016, 1, 30),
    ) -> None:
        """
        Args:
            tickers: tickers
            start: start time
            end: end time
        """
        super().__init__()
        self._tickers = []
        tickers = tickers if tickers is not None else []
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace("\n", ";").split(";")
        self._n = len(self._tickers)

        self._start = start.strftime("%Y-%m-%d")
        self._end = end.strftime("%Y-%m-%d")
        self._data = []

    def run(self) -> None:
        """
        Loads data, thus enabling get_similarity_matrix and
        get_covariance_matrix methods in the base class.
        """
        if len(self._tickers) == 0:
            raise QiskitFinanceError("Missing tickers to download.")
        self._data = []
        stocks_notfound = []
        try:
            # download multiple tickers in single thread to avoid
            # race condition
            stock_data = yf.download(
                self._tickers,
                start=self._start,
                end=self._end,
                group_by="ticker",
                threads=False,
                progress=logger.isEnabledFor(logging.DEBUG),
            )
            if len(self._tickers) == 1:
                ticker_name = self._tickers[0]
                stock_value = stock_data["Adj Close"]
                if stock_value.dropna().empty:
                    stocks_notfound.append(ticker_name)
                self._data.append(stock_value)
            else:
                for ticker_name in self._tickers:
                    stock_value = stock_data[ticker_name]["Adj Close"]
                    if stock_value.dropna().empty:
                        stocks_notfound.append(ticker_name)
                    self._data.append(stock_value)
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug(ex, exc_info=True)
            raise QiskitFinanceError("Accessing Yahoo Data failed.") from ex

        if stocks_notfound:
            raise QiskitFinanceError(
                f"No data found for this date range, symbols may be delisted: {stocks_notfound}."
            )

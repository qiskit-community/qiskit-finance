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

""" Exchange data provider. """

from __future__ import annotations
import logging
import datetime
import nasdaqdatalink

from ._base_data_provider import BaseDataProvider, StockMarket
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)

VALID_STOCKMARKETS = [
    StockMarket.LONDON,
    StockMarket.EURONEXT,
    StockMarket.SINGAPORE,
]


class ExchangeDataProvider(BaseDataProvider):
    """Exchange data provider.

    Please see:
    https://qiskit-community.github.io/qiskit-finance/tutorials/11_time_series.html
    for instructions on use, which involve obtaining a Nasdaq Data Link access token.
    """

    def __init__(
        self,
        token: str,
        tickers: str | list[str] | None = None,
        stockmarket: StockMarket = StockMarket.LONDON,
        start: datetime.datetime = datetime.datetime(2016, 1, 1),
        end: datetime.datetime = datetime.datetime(2016, 1, 30),
    ) -> None:
        """
        Args:
            token (str): Nasdaq Data Link access token.
            tickers (str | list[str] | None): Tickers for the data provider.

                * If a string is provided, it can be a single ticker symbol or multiple symbols
                  separated by semicolons or newlines.
                * If a list of strings is provided, each string should be a single ticker symbol.

                Default is :code:`None`, which corresponds to no tickers provided.
            stockmarket (StockMarket): LONDON (default), EURONEXT, or SINGAPORE
            start (datetime.datetime): Start date of the data.
                Defaults to January 1st, 2016.
            end (datetime.datetime): End date of the data.
                Defaults to January 30th, 2016.

        Raises:
            QiskitFinanceError: provider doesn't support given stock market.
        """
        super().__init__()

        if tickers is None:
            tickers = []
        if isinstance(tickers, str):
            tickers = tickers.replace("\n", ";").split(";")

        self._tickers = tickers
        self._n = len(tickers)

        if stockmarket not in VALID_STOCKMARKETS:
            msg = f"ExchangeDataProvider does not support {stockmarket.value} as a stock market."
            raise QiskitFinanceError(msg)

        # This is to aid serialization; string is ok to serialize
        self._stockmarket = str(stockmarket.value)

        self._token = token
        self._tickers = tickers
        self._start = start.strftime("%Y-%m-%d")
        self._end = end.strftime("%Y-%m-%d")

    def run(self) -> None:
        """
        Loads data, thus enabling :meth:`get_similarity_matrix` and :meth:`get_covariance_matrix`
        methods in the base class.
        """
        nasdaqdatalink.ApiConfig.api_key = self._token
        self._data = []
        stocks_notfound = []
        stocks_forbidden = []
        for _, ticker_name in enumerate(self._tickers):
            stock_data = None
            name = self._stockmarket + "/" + ticker_name
            try:
                stock_data = nasdaqdatalink.get(name, start_date=self._start, end_date=self._end)
            except nasdaqdatalink.AuthenticationError as ex:
                logger.debug(ex, exc_info=True)
                raise QiskitFinanceError("Nasdaq Data Link invalid token.") from ex
            except nasdaqdatalink.LimitExceededError as ex:
                logger.debug(ex, exc_info=True)
                raise QiskitFinanceError("Nasdaq Data Link limit exceeded.") from ex
            except nasdaqdatalink.NotFoundError as ex:
                logger.debug(ex, exc_info=True)
                stocks_notfound.append(name)
                continue
            except nasdaqdatalink.ForbiddenError as ex:
                logger.debug(ex, exc_info=True)
                stocks_forbidden.append(name)
                continue
            except nasdaqdatalink.DataLinkError as ex:
                logger.debug(ex, exc_info=True)
                raise QiskitFinanceError(f"Nasdaq Data Link Error for '{name}'.") from ex
            try:
                self._data.append(stock_data["Close"])
            except KeyError as ex:
                raise QiskitFinanceError(f"Cannot parse Nasdaq Data Link '{name}' output.") from ex

        if stocks_notfound or stocks_forbidden:
            err_msg = f"Stocks not found: {stocks_notfound}. " if stocks_notfound else ""
            if stocks_forbidden:
                err_msg += (
                    "You do not have permission to view this data. "
                    f"Please subscribe to this database: {stocks_forbidden}"
                )
            raise QiskitFinanceError(err_msg)

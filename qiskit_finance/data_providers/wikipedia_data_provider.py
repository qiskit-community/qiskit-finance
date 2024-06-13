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

"""Wikipedia data provider."""

from __future__ import annotations
import logging
import datetime
import nasdaqdatalink

from ._base_data_provider import BaseDataProvider
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


class WikipediaDataProvider(BaseDataProvider):
    """Wikipedia data provider.

    This data provider retrieves stock market data from the Wikipedia dataset
    using Nasdaq Data Link API. For more details on usage, please refer to the
    official documentation:
    https://qiskit-community.github.io/qiskit-finance/tutorials/11_time_series.html
    """

    def __init__(
        self,
        token: str | None = None,
        tickers: str | list[str] | None = None,
        start: datetime.datetime = datetime.datetime(2016, 1, 1),
        end: datetime.datetime = datetime.datetime(2016, 1, 30),
    ) -> None:
        """
        Initialize the Wikipedia Data Provider.

        Args:
            token (str | None): Nasdaq Data Link access token.
                Default is None.
            tickers (str | list[str] | None): Tickers for the data provider.

                * If a string is provided, it can be a single ticker symbol or multiple symbols
                  separated by semicolons or newlines.
                * If a list of strings is provided, each string should be a single ticker symbol.

                Default is :code:`None`, which corresponds to no tickers provided.
            start (datetime.datetime): Start date of the data.
                Default is January 1st, 2016.
            end (datetime.datetime): End date of the data.
                Default is January 30th, 2016.
        """
        super().__init__()

        if tickers is None:
            tickers = []
        if isinstance(tickers, str):
            tickers = tickers.replace("\n", ";").split(";")

        self._tickers = tickers
        self._n = len(tickers)
        self._token = token
        self._start = start.strftime("%Y-%m-%d")
        self._end = end.strftime("%Y-%m-%d")
        self._data = []

    def run(self) -> None:
        """
        Loads data from Wikipedia using Nasdaq Data Link API.
        Retrieves stock market data from the Wikipedia dataset
        using Nasdaq Data Link API, and populates the data attribute of the
        base class, enabling further calculations like similarity and covariance
        matrices.

        Raises:
            QiskitFinanceError: If there is an invalid Nasdaq Data Link token,
                if the Nasdaq Data Link limit is exceeded, if data is not found
                for the specified tickers, or if there is an error accessing
                Nasdaq Data Link.
        """
        nasdaqdatalink.ApiConfig.api_key = self._token
        self._data = []
        stocks_notfound = []
        for _, ticker_name in enumerate(self._tickers):
            stock_data = None
            name = "WIKI" + "/" + ticker_name
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
            except nasdaqdatalink.DataLinkError as ex:
                logger.debug(ex, exc_info=True)
                raise QiskitFinanceError(f"Nasdaq Data Link Error for '{name}'.") from ex

            try:
                self._data.append(stock_data["Adj. Close"])
            except KeyError as ex:
                raise QiskitFinanceError("Cannot parse Nasdaq Data Link output.") from ex

        if stocks_notfound:
            raise QiskitFinanceError(f"Stocks not found: {stocks_notfound}. ")

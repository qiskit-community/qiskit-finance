# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Wikipedia data provider. """

from typing import Optional, Union, List
import logging
import datetime
import nasdaqdatalink

from ._base_data_provider import BaseDataProvider
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


class WikipediaDataProvider(BaseDataProvider):
    """Wikipedia data provider.

    Please see:
    https://github.com/Qiskit/qiskit-finance/blob/main/docs/tutorials/11_time_series.ipynb
    for instructions on use.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        tickers: Optional[Union[str, List[str]]] = None,
        start: datetime.datetime = datetime.datetime(2016, 1, 1),
        end: datetime.datetime = datetime.datetime(2016, 1, 30),
    ) -> None:
        """
        Args:
            token: Nasdaq Data Link access token, which is not needed, strictly speaking
            tickers: tickers
            start: start time
            end: end time
        """
        super().__init__()
        self._tickers = None  # type: Optional[Union[str, List[str]]]
        tickers = tickers if tickers is not None else []
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace("\n", ";").split(";")
        self._n = len(self._tickers)

        self._token = token
        self._tickers = tickers
        self._start = start.strftime("%Y-%m-%d")
        self._end = end.strftime("%Y-%m-%d")
        self._data = []

    def run(self) -> None:
        """
        Loads data, thus enabling get_similarity_matrix and
        get_covariance_matrix methods in the base class.
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

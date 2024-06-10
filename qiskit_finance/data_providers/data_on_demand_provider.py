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

""" NASDAQ Data on demand data provider. """

from __future__ import annotations
import datetime
from urllib.parse import urlencode
import logging
import json
import certifi
import urllib3

from ._base_data_provider import BaseDataProvider
from ..exceptions import QiskitFinanceError

logger = logging.getLogger(__name__)


class DataOnDemandProvider(BaseDataProvider):
    """NASDAQ Data on Demand data provider.

    Please see:
    https://qiskit-community.github.io/qiskit-finance/tutorials/11_time_series.html
    for instructions on use, which involve obtaining a NASDAQ DOD access token.
    """

    def __init__(
        self,
        token: str,
        tickers: str | list[str] | None = None,
        start: datetime.datetime = datetime.datetime(2016, 1, 1),
        end: datetime.datetime = datetime.datetime(2016, 1, 30),
        verify: str | bool | None = None,
    ) -> None:
        """
        Args:
            token (str): Nasdaq Data Link access token.
            tickers (str | list[str] | None): Tickers for the data provider.
                - If a string is provided, it can be a single ticker symbol or multiple symbols
                  separated by semicolons or newlines.
                - If a list of strings is provided, each string should be a single ticker symbol.
                Default is None.
            start (datetime.datetime): Start date of the data.
                Defaults to January 1st, 2016.
            end (datetime.datetime): End date of the data.
                Defaults to January 30th, 2016.
            verify (str | bool | None): If verify is None, certify certificates
                will be used (default);
                if this is False, no certificates will be checked; if this is a string,
                it should be pointing
                to a certificate for the HTTPS connection to NASDAQ (`dataondemand.nasdaq.com`),
                either in the
                form of a CA_BUNDLE file or a directory wherein to look.
        """
        super().__init__()

        self._tickers = None
        tickers = tickers if tickers is not None else []
        if isinstance(tickers, list):
            self._tickers = tickers
        else:
            self._tickers = tickers.replace("\n", ";").split(";")
        self._n = len(self._tickers)

        self._token = token
        self._start = start
        self._end = end
        self._verify = verify

    def run(self) -> None:
        """
        Loads data, thus enabling get_similarity_matrix and get_covariance_matrix
        methods in the base class.
        """

        http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
        url = "https://dataondemand.nasdaq.com/api/v1/quotes?"
        self._data = []
        stocks_error = []
        try:
            for ticker in self._tickers:
                values = {
                    "_Token": self._token,
                    "symbols": [ticker],
                    "start": self._start.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"),
                    "end": self._end.strftime("%Y-%m-%d'T'%H:%M:%S.%f'Z'"),
                    "next_cursor": 0,
                }
                encoded = url + urlencode(values)
                if self._verify is None:
                    response = http.request(
                        "POST", encoded
                    )  # this runs certificate verification, as per the set-up of the urllib3
                else:
                    # this disables certificate verification (False)
                    # or forces the certificate path (str)
                    response = http.request("POST", encoded, verify=self._verify)
                if response.status != 200:
                    logger.debug(response.data.decode("utf-8"))
                    stocks_error.append(ticker)
                    continue
                quotes = json.loads(response.data.decode("utf-8"))["quotes"]
                price_evolution = []
                for q in quotes:
                    price_evolution.append(q["ask_price"])
                self._data.append(price_evolution)
        finally:
            http.clear()

        if stocks_error:
            err_msg = (
                f"Accessing NASDAQ Data on Demand with symbols: {stocks_error}, "
                f"start: {self._start}, end: {self._end} failed. "
                "Hint: Check the token. Check the spelling of ticker."
            )
            raise QiskitFinanceError(err_msg)

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

""" Test Data Providers """

import unittest
import warnings
import os
import datetime
from test import QiskitFinanceTestCase
from ddt import ddt, data, unpack
import nasdaqdatalink
import numpy as np
from qiskit_finance import QiskitFinanceError
from qiskit_finance.data_providers import (
    RandomDataProvider,
    WikipediaDataProvider,
    YahooDataProvider,
    StockMarket,
    DataOnDemandProvider,
    ExchangeDataProvider,
)


@ddt
class TestDataProviders(QiskitFinanceTestCase):
    """Tests data providers for the Portfolio Optimization and Diversification."""

    logger = None

    def setUp(self):
        super().setUp()
        self._nasdaq_data_link_api_key = (
            os.getenv("NASDAQ_DATA_LINK_API_KEY") if os.getenv("NASDAQ_DATA_LINK_API_KEY") else None
        )
        self._on_demand_token = (
            os.getenv("ON_DEMAND_TOKEN") if os.getenv("ON_DEMAND_TOKEN") else None
        )
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
        warnings.filterwarnings(action="ignore", module="urllib3", category=DeprecationWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="default", message="unclosed", category=ResourceWarning)
        warnings.filterwarnings(action="default", module="urllib3", category=DeprecationWarning)

    def test_random_wrong_use(self):
        """Random wrong use test"""
        rnd = RandomDataProvider(seed=1)
        # Now, the .run() method is expected, which does the actual data loading
        # (and can take seconds or minutes,
        # depending on the data volumes, hence not ok in the constructor)
        with self.subTest("test RandomDataProvider get_covariance_matrix"):
            self.assertRaises(QiskitFinanceError, rnd.get_covariance_matrix)
        with self.subTest("test RandomDataProvider get_similarity_matrix"):
            self.assertRaises(QiskitFinanceError, rnd.get_similarity_matrix)
        wiki = WikipediaDataProvider(
            token=self._nasdaq_data_link_api_key,
            tickers=["GOOG", "AAPL"],
            start=datetime.datetime(2016, 1, 1),
            end=datetime.datetime(2016, 1, 30),
        )
        # Now, the .run() method is expected, which does the actual data loading
        with self.subTest("test WikipediaDataProvider get_covariance_matrix"):
            self.assertRaises(QiskitFinanceError, wiki.get_covariance_matrix)
        with self.subTest("test WikipediaDataProvider get_similarity_matrix"):
            self.assertRaises(QiskitFinanceError, wiki.get_similarity_matrix)

    def test_yahoo_wrong_use(self):
        """Yahoo! wrong use test"""
        yahoo = YahooDataProvider(
            tickers=["AEO", "ABBY"],
            start=datetime.datetime(2018, 1, 1),
            end=datetime.datetime(2018, 12, 31),
        )
        # Now, the .run() method is expected, which does the actual data loading
        with self.subTest("test YahooDataProvider get_covariance_matrix"):
            self.assertRaises(QiskitFinanceError, yahoo.get_covariance_matrix)
        with self.subTest("test YahooDataProvider get_similarity_matrix"):
            self.assertRaises(QiskitFinanceError, yahoo.get_similarity_matrix)

    def test_random(self):
        """random test"""
        similarity = np.array([[1.00000000e00, 6.2284804e-04], [6.2284804e-04, 1.00000000e00]])
        covariance = np.array([[2.08413157, 0.20842107], [0.20842107, 1.99542187]])
        rnd = RandomDataProvider(seed=1)
        rnd.run()
        with self.subTest("test RandomDataProvider get_covariance_matrix"):
            np.testing.assert_array_almost_equal(rnd.get_covariance_matrix(), covariance, decimal=3)
        with self.subTest("test RandomDataProvider get_similarity_matrix"):
            np.testing.assert_array_almost_equal(rnd.get_similarity_matrix(), similarity, decimal=3)

    def test_random_divide_0(self):
        """Random divide by 0 test"""
        # This will create data with some 0 values, it should not throw
        # divide by 0 errors
        seed = 8888
        num_assets = 4
        stocks = [f"TICKER{i}" for i in range(num_assets)]
        random_data = RandomDataProvider(
            tickers=stocks,
            start=datetime.datetime(2016, 1, 1),
            end=datetime.datetime(2016, 1, 30),
            seed=seed,
        )
        random_data.run()
        mu_value = random_data.get_period_return_mean_vector()
        sigma_value = random_data.get_period_return_covariance_matrix()
        with self.subTest("test get_period_return_mean_vector is numpy array"):
            self.assertIsInstance(mu_value, np.ndarray)
        with self.subTest("test get_period_return_covariance_matrix is numpy array"):
            self.assertIsInstance(sigma_value, np.ndarray)

    def test_wikipedia(self):
        """wikipedia test"""
        try:
            wiki = WikipediaDataProvider(
                token=self._nasdaq_data_link_api_key,
                tickers=["GOOG", "AAPL"],
                start=datetime.datetime(2016, 1, 1),
                end=datetime.datetime(2016, 1, 30),
            )
            wiki.run()
            similarity = np.array(
                [[1.00000000e00, 8.44268222e-05], [8.44268222e-05, 1.00000000e00]]
            )
            covariance = np.array([[269.60118129, 25.42252332], [25.42252332, 7.86304499]])
            with self.subTest("test WikipediaDataProvider get_covariance_matrix"):
                np.testing.assert_array_almost_equal(
                    wiki.get_covariance_matrix(), covariance, decimal=3
                )
            with self.subTest("test WikipediaDataProvider get_similarity_matrix"):
                np.testing.assert_array_almost_equal(
                    wiki.get_similarity_matrix(), similarity, decimal=3
                )
        except QiskitFinanceError as ex:
            if isinstance(ex.__cause__, nasdaqdatalink.LimitExceededError):
                self.skipTest(f"Test of WikipediaDataProvider skipped: {str(ex)}")
            else:
                self.fail(f"Test of WikipediaDataProvider failed: {str(ex)}")

    def test_nasdaq(self):
        """nasdaq test"""
        if self._on_demand_token is None:
            self.skipTest(
                "Skipped NASDAQ On Demand test since token was not provided in "
                "env. variable ON_DEMAND_TOKEN"
            )
            return
        try:
            nasdaq = DataOnDemandProvider(
                token=self._on_demand_token,
                tickers=["GOOG", "AAPL"],
                start=datetime.datetime(2016, 1, 1),
                end=datetime.datetime(2016, 1, 2),
            )
            nasdaq.run()
        except QiskitFinanceError as ex:
            self.fail(f"Test of DataOnDemandProvider failed: {str(ex)}")

    def test_exchangedata(self):
        """exchange data test"""
        if self._nasdaq_data_link_api_key is None:
            self.skipTest(
                "Skipped NASDAQ Data Link test since token was not provided in "
                "env. variable NASDAQ_DATA_LINK_API_KEY"
            )
            return
        try:
            lse = ExchangeDataProvider(
                token=self._nasdaq_data_link_api_key,
                tickers=["AEO", "ABBY"],
                stockmarket=StockMarket.LONDON,
                start=datetime.datetime(2018, 1, 1),
                end=datetime.datetime(2018, 12, 31),
            )
            lse.run()
            similarity = np.array(
                [[1.00000000e00, 8.44268222e-05], [8.44268222e-05, 1.00000000e00]]
            )
            covariance = np.array([[2.693, -18.65], [-18.65, 1304.422]])
            with self.subTest("test ExchangeDataProvider get_covariance_matrix"):
                np.testing.assert_array_almost_equal(
                    lse.get_covariance_matrix(), covariance, decimal=3
                )
            with self.subTest("test ExchangeDataProvider get_similarity_matrix"):
                np.testing.assert_array_almost_equal(
                    lse.get_similarity_matrix(), similarity, decimal=3
                )
        except QiskitFinanceError as ex:
            self.fail(f"Test of ExchangeDataProvider failed: {str(ex)}")

    @data(
        [["AEO", "AEP"], [[7.0, 1.0], [1.0, 15.0]], [[1.0e00, 9.2e-05], [9.2e-05, 1.0e00]]],
        ["AEO", 7.0, [[1.0]]],
    )
    @unpack
    def test_yahoo(self, tickers, covariance, similarity):
        """Yahoo data test"""
        yahoo = YahooDataProvider(
            tickers=tickers,
            start=datetime.datetime(2018, 1, 1),
            end=datetime.datetime(2018, 12, 31),
        )
        try:
            yahoo.run()
            with self.subTest("test YahooDataProvider get_covariance_matrix"):
                np.testing.assert_array_almost_equal(
                    yahoo.get_covariance_matrix(), np.array(covariance), decimal=0
                )
            with self.subTest("test YahooDataProvider get_similarity_matrix"):
                np.testing.assert_array_almost_equal(
                    yahoo.get_similarity_matrix(), np.array(similarity), decimal=1
                )
        except QiskitFinanceError as ex:
            self.fail(f"Test of YahooDataProvider failed: {str(ex)}")


if __name__ == "__main__":
    unittest.main()

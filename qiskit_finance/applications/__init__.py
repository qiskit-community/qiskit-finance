# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Finance applications (:mod:`qiskit_finance.applications`)
=========================================================

.. currentmodule:: qiskit_finance.applications

Applications for Qiskit's finance module.

Optimization Applications
=========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PortfolioOptimization
   PortfolioDiversification

Estimation Applications
=======================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EuropeanCallDelta
   EuropeanCallPricing
   FixedIncomePricing


"""

from .estimation import EuropeanCallDelta, EuropeanCallPricing, FixedIncomePricing
from .optimization import PortfolioOptimization, PortfolioDiversification

__all__ = [
    "PortfolioOptimization",
    "PortfolioDiversification",
    "EuropeanCallDelta",
    "EuropeanCallPricing",
    "FixedIncomePricing",
]

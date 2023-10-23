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

"""
Finance applications (:mod:`qiskit_finance.applications`)
=========================================================

.. currentmodule:: qiskit_finance.applications

Qiskit Finance ready-made applications.

Optimization Applications
-------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PortfolioOptimization
   PortfolioDiversification

Estimation Applications
-----------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EstimationApplication
   EuropeanCallDelta
   EuropeanCallPricing
   FixedIncomePricing


"""

from .estimation import (
    EstimationApplication,
    EuropeanCallDelta,
    EuropeanCallPricing,
    FixedIncomePricing,
)
from .optimization import PortfolioOptimization, PortfolioDiversification

__all__ = [
    "PortfolioOptimization",
    "PortfolioDiversification",
    "EstimationApplication",
    "EuropeanCallDelta",
    "EuropeanCallPricing",
    "FixedIncomePricing",
]

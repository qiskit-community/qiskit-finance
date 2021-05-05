# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
===============================================
Qiskit's finance module (:mod:`qiskit_finance`)
===============================================

.. currentmodule:: qiskit_finance

This is the Qiskit`s finance module. There is an initial set of function here that
will be built out over time. At present it has applications in the form of
Ising Hamiltonians and data providers which supply a source of financial data.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QiskitFinanceError

In addition to standard Python errors Qiskit's finance module will raise this error if
circumstances are that it cannot proceed to completion.

Submodules
==========

.. autosummary::
   :toctree:

   applications
   data_providers

"""

from .version import __version__
from .exceptions import QiskitFinanceError

__all__ = ["__version__", "QiskitFinanceError"]

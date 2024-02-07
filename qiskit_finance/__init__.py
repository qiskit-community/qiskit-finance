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
"""
=============================================
Qiskit Finance module (:mod:`qiskit_finance`)
=============================================

.. currentmodule:: qiskit_finance

This is the Qiskit Finance module. It has applications based on
`Amplitude Estimation
<https://qiskit-community.github.io/qiskit-algorithms/apidocs/qiskit_algorithms.html#amplitude-estimators>`__
and optimization using
`Qiskit Optimization <https://qiskit-community.github.io/qiskit-optimization/>`__,
some library circuits useful for finance applications,
and data providers which supply a source of financial data.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QiskitFinanceError

In addition to standard Python errors the Qiskit Finance module will raise this error
if circumstances are that it cannot proceed to completion.

Submodules
==========

.. autosummary::
   :toctree:

   applications
   circuit
   data_providers

"""

from .version import __version__
from .exceptions import QiskitFinanceError

__all__ = ["__version__", "QiskitFinanceError"]

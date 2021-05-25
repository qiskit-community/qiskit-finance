:orphan:

###############
Getting started
###############

Installation
============

Qiskit Finance depends on the main Qiskit package which has its own
`Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__ detailing the
installation options for Qiskit and its supported environments/platforms. You should refer to
that first. Then the information here can be followed which focuses on the additional installation
specific to Qiskit Finance.

.. tabbed:: Start locally

    The simplest way to get started is to follow the getting started 'Start locally' for Qiskit
    here `Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__

    In your virtual environment where you installed Qiskit simply add ``finance`` to the
    extra list in a similar manner to how the extra ``visualization`` support is installed for
    Qiskit, i.e:

    .. code:: sh

        pip install qiskit[finance]

    It is worth pointing out that if you're a zsh user (which is the default shell on newer
    versions of macOS), you'll need to put ``qiskit[finance]`` in quotes:

    .. code:: sh

        pip install 'qiskit[finance]'


.. tabbed:: Install from source

   Installing Qiskit Finance from source allows you to access the most recently
   updated version under development instead of using the version in the Python Package
   Index (PyPI) repository. This will give you the ability to inspect and extend
   the latest version of the Qiskit Finance code more efficiently.

   Since Qiskit Finance depends on Qiskit, and its latest changes may require new or changed
   features of Qiskit, you should first follow Qiskit's `"Install from source"` instructions
   here `Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__

   .. raw:: html

      <h2>Installing Qiskit Finance from Source</h2>

   Using the same development environment that you installed Qiskit in you are ready to install
   Qiskit Finance.

   1. Clone the Qiskit Finance repository.

      .. code:: sh

         git clone https://github.com/Qiskit/qiskit-finance.git

   2. Cloning the repository creates a local folder called ``qiskit-finance``.

      .. code:: sh

         cd qiskit-finance

   3. If you want to run tests or linting checks, install the developer requirements.

      .. code:: sh

         pip install -r requirements-dev.txt

   4. Install ``qiskit-finance``.

      .. code:: sh

         pip install .

   If you want to install it in editable mode, meaning that code changes to the
   project don't require a reinstall to be applied, you can do this with:

   .. code:: sh

      pip install -e .

----

Ready to get going?...
======================

.. raw:: html

   <div class="tutorials-callout-container">
      <div class="row">

.. customcalloutitem::
   :description: Find out about Qiskit Finance.
   :header: Dive into the tutorials
   :button_link:  ./tutorials/index.html
   :button_text: Qiskit Finance tutorials

.. raw:: html

      </div>
   </div>


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`

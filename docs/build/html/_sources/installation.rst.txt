Installation
============

Install pyworld3 using `pip`:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/mBarreau/pyworld3

Run the provided example to simulate the standard run, known as the *Business
as usual* scenario:

.. code-block:: python

   import pyworld3
   pyworld3.hello_world3()

As shown below, the simulation output compares well with the original print.
For a tangible understanding by the general audience, the usual chart plots the
trajectories of the:

* population (`POP`) from the Population sector,
* nonrenewable resource fraction remaining (`NRFR`) from the Nonrenewable Resource sector,
* food per capita (`FPC`) from the Agriculture sector,
* industrial output per capita (`IOPC`) from the Capital sector,
* index of persistent pollution (`PPOLX`) from the Persistent Pollution sector.

.. image:: ../../img/result_standard_run.png

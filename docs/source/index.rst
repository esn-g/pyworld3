.. PyWorld3 documentation master file, created by
   sphinx-quickstart on Fri Jan  5 15:12:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyWorld3's documentation!
====================================

PyWorld3 is a Python implementation of the World3 model, as described in
the book *Dynamics of Growth in a Finite World*. This version slightly differs
from the previous one used in the world-known reference *the Limits to Growth*,
because of different numerical parameters and a slightly different model
structure.

The World3 model is based on an Ordinary Differential Equation solved by a
Backward Euler method. Although it is described with 12 state variables, taking
internal delay functions into account raises the problem to the 29th order. For
the sake of clarity and model calibration purposes, the model is structured
into 5 main sectors: Population, Capital, Agriculture, Persistent Pollution
and Nonrenewable Resource. For a better understanding of variables and their
relations, one can refer to `this <https://abaucher.gitlabpages.inria.fr/pydynamo/w3_sectors.html>`_.

.. toctree::
   installation
   API


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

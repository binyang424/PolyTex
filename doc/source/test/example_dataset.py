# ！/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example dataset
===============

We provide several example datasets for testing and demonstration purposes.
This file shows how to download and load the example datasets. By default,
the datasets are stored in the user's current working directory in a folder
called ``test_data``. This can be changed by specifying the path as demonstrated
below.
"""

###############################################################################
# Check the available datasets
# -------------------------------------
import polytex as pk
pk.example(data_name="all")

###############################################################################
# Downloading the example datasets
# -------------------------------------
#
# The example datasets are hosted on a public Github repository.
# To download the datasets, use the ``polytex.example()`` function.
# This function will download the datasets to the user's current working
# directory in a folder called ``test_data``. This can be changed by specifying
# the path using the entry ``outdir``.

pk.example("surface points", outdir="./test_data/")

###############################################################################
# The input and output of the example datasets will be described in the examples
# for ``polytex.io`` module.

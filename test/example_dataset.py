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

import polykriging as pk

###############################################################################
# Downloading the example datasets
# --------------------------------
#
# The example datasets are hosted on a public github repository.
# To download the datasets, use the ``polykriging.example()`` function.
# This function will download the datasets to the user's current working
# directory in a folder called ``test_data``. This can be changed by specifying
# the path.

pk.example("git")


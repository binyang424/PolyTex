"""
polykriging.

An implementation of dual Kriging.
"""

__version__ = "0.1"
__author__ = 'Bin Yang'
__credits__ = 'Polytechnique Montreal & Wuhan University of Technology'


from .fileio import *
from . import geometry
from . import stats

from . import thirdparty
from .plot import *
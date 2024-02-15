"""
polytex.

An implementation of dual kriging.
"""

__version__ = "0.1"
__author__ = 'Bin Yang'
__credits__ = 'Polytechnique Montreal & Wuhan University of Technology'

from . import bw_opt
from .bw_opt import *

from . import kde
from .kde import *

from .correlation import *

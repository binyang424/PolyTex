"""
PolyTex.
==================
An implementation of dual kriging.
"""

__version__ = "0.4.4"
__author__ = 'Bin Yang'
__credits__ = 'Polytechnique Montreal & Wuhan University of Technology'


from . import geometry
from . import kriging
from . import mesh
from . import stats
from . import thirdparty
from . import plot

# from .fileio import *
from .textile import *
from .tow import *

from .__dataset__ import example, __download_file

from .io import *
from . import io


from .misc import *
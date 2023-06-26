from .io import pk_save, pk_load, save_ply
from .geometry import geom_tow, Tube, ParamCurve
from .kriging import curve2Dinter, surface3Dinterp
from .stats import kdeScreen, bw_scott
import numpy as np


class Textile:
    """
    This is a class for textile geometrical analysis and numerical meshing.
    定义为体素网格？

    # TODO : analyze, design, and program the Textile class.
    """

    def __init__(self, x, y, z, name="textile"):
        """
        Initialize the textile object by creating a background mesh.
        """
        self.name = name
        self.background_mesh(x, y, z)

    def add(self, tow):
        """
        Add a tow to the textile using mesh merging. Each tow is a mesh object
        and labeled with a unique integer.

        key - tow id, integer;
        value - tow object
        """

        pass

    def remove(self, tow):
        pass

    def triangulate(self):
        """
        Hexahedral mesh to tetrahedral mesh. Conformal meshing.
        """

        pass

    def decimate(self):
        """
        Decimate the mesh.
        """
        pass

    def background_mesh(self, x, y, z):
        """
        Generate a background mesh for the textile.
        """
        pass

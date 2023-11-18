from .io import pk_save, pk_load, filenames, choose_directory
from .tow import Tow
import numpy as np
from .mesh import background_mesh, label_mask, intersection_detect
import pyvista as pv
from tqdm.auto import tqdm
from scipy.sparse import coo_matrix

import re


class Textile:
    """
    This is a class for textile geometrical analysis and numerical meshing.

    # TODO : analyze, design, and program the Textile class.
    """

    def __init__(self, name="textile"):
        """
        Initialize the textile object by creating a background mesh.
        """
        self.name = name
        self.tows = {}
        self.groups = {}

    def add_tow(self, tow, group=None):
        """
        Add a tow to the textile. If tow is already in the textile, then raise
        a ValueError.

        Parameters
        ----------
        tow : Tow object
            Tow to be added to the textile. Stored in self.tows as a dictionary.
            Tow.name is the key, and tow is the value.
        group : str
            Group name of the tow. If group is None, then the tow is not added
            to any group. Stored in self.groups.

        Returns
        -------
        None.
        """
        if not isinstance(tow, Tow):
            raise TypeError("Input tow must be a Tow object.")

        if tow.name in self.tows.keys():
            raise ValueError("Tow already exists. Please use"
                             "another name for the new tow.")

        self.tows[tow.name] = tow

        if group is not None:
            self.add_group(name=group, tow=tow)

    def add_group(self, name="group1", tow=None):
        """
        Groups of the tows in the textile. Each is a group of Tow objects.
        if tow is None, then the group is empty.

        Parameters
        ----------
        name : str
            Name of the group.
        tow : Tow object
            Tow to be added to the group. If tow is None, then the group is empty.
            If tow is not None, then tow is added to the group. Besides, if tow is
            not in the self.tows yet, it will be added to self.tows.

        Returns
        -------
        None.
        """
        if tow is None:
            if name in self.groups.keys():
                raise ValueError("Group already exists. Please use another name for the "
                                 "new group.")
            else:
                self.groups[name] = []
                return None

        if not isinstance(tow, Tow):
            raise TypeError("Input tow must be a Tow object.")

        if tow.name not in self.tows.keys():
            self.add_tow(tow)

        if name not in self.groups.keys():
            self.groups[name] = []
            self.groups[name].append(tow.name)
        elif tow.name not in self.groups[name]:
            self.groups[name].append(tow.name)
        else:
            raise ValueError("Tow already exists in the group.")

        return None

    def remove(self, tow):
        """
        Remove a tow from the textile.

        Parameters
        ----------
        tow : str or Tow object
            The tow to be removed.

        Returns
        -------
        None.
        """
        if isinstance(tow, Tow):
            tow = tow.name

        if tow not in self.tows.keys():
            raise ValueError("Tow does not exist.")

        self.tows.pop(tow)

        print("Tow %s is removed." % tow)

        # check if the tow is in any group
        if self.groups != {}:
            for group in self.groups.keys():
                if tow in groups[group]:
                    group.pop(tow)
                    print("Tow %s is removed from group %s." % (tow, group))

        return None

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

    def mesh(self, bbox, voxel_size=None, show=False):
        """
        Generate a mesh for the textile.

        Parameters
        ----------
        bbox: bounding box of the background mesh specified through a numpy array
            contains the minimum and maximum coordinates of the bounding box
            [xmin, xmax, ymin, ymax, zmin, zmax]
        voxel_size : float, or numpy.ndarray
            voxel size of the background mesh.
                if `None`, the voxel size is set to the 1/20 of the diagonal length of the bounding box;
                if `float`, the voxel size is set to the float value in x, y, z directions;
                if `list`, `set`, or `tuple` of size 3, the voxel size is set to the values for
                the x-, y- and z- directions.

        Returns
        -------
        grid : pyvista mesh object (UnstructuredGrid)
        mesh_shape : tuple
            shape of the mesh
        """
        mesh_background, mesh_shape = background_mesh(bbox, voxel_size=voxel_size)
        self.mesh = mesh_background
        self.mesh_shape = mesh_shape
        self.n_cells = mesh_background.n_cells

        if show:
            self.mesh.plot(show_edges=True)

    def cell_labeling(self, intersection=False, check_surface=False):
        """
        Label the cells of the background mesh with tow id.

        Parameters
        ----------
        intersection : bool, optional
            Whether to detect the intersection of the tows. The default is False.
        """

        # initialize the array of label list with -1
        label_list = np.full(self.n_cells, -1, dtype=np.int32)

        """ Select the surface meshes of yarns to be labelled """
        label_set_dict = dict()

        path = choose_directory("Choose the surface mesh directory for fiber tow labelling")
        file_list = filenames(path, ".stl")
        file_list_sort = {}
        for i, file in enumerate(file_list):
            # regular expression for integer
            yarn_index = re.findall(r'\d+', file)
            file_list_sort[int(yarn_index[0])] = file

        print("Labeling cells of the background mesh with tow id...")

        """ Label the surface meshes of yarns """
        indices = np.array(list(file_list_sort.keys()))
        indices.sort()

        for index in indices:
            ## `check_surface=False`
            # if index in [28, 31, 57]:
            #     print("Skipping yarn %d of %d: %s" % (index, len(yarnIndex), file_list_sort[index]))
            #     continue

            print("Processing yarn %d of %d: %s" % (index + 1, len(indices), file_list_sort[index]))

            mesh_tri = pv.read(path + "\\" + file_list_sort[index])  # load surface mesh

            # find the cells that are within the tubular surface of the fiber tow
            mask, label_yarn = label_mask(self.mesh, mesh_tri, check_surface=False)

            label_list[mask] = index
            label_set_dict[index] = coo_matrix(label_yarn)

        self.mesh.cell_data['yarnIndex'] = label_list

        if intersection:
            """ Label the intersected cells"""
            label_list = intersection_detect(label_list, label_set_dict)

            intersect_info, intersect_info_dict, cell_data_intersect = intersection_detect(label_set_dict)
            self.mesh.cell_data['intersection'] = cell_data_intersect

        return None

